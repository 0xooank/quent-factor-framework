from jqdata import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.optimize as opt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 完整策略实现


def initialize(context):
    """策略初始化"""
    # 基础设置
    g.benchmark = '000300.XSHG'
    g.stock_pool = get_index_stocks('000300.XSHG')
    # 因子配置（价值+质量+成长组合）
    g.factors = {
        'bp_ratio': 'bp_ratio',      # 市净率倒数（价值）
        'roe': 'roe',                      # ROE（质量）
        'roa': 'roa',                      # ROA（质量）
        'gross_margin': 'gross_profit_margin',  # 毛利率（质量）
        'inc_revenue': 'inc_revenue_year_on_year'  # 营收增长（成长）
    }

    # 策略参数
    g.position_count = 50
    g.max_single_weight = 0.03
    g.rebalance_days = 11

    # 风险约束（Barra风格）
    g.risk_constraints = {
        'size': (-0.5, 0.5),
        'value': (-1.0, 1.5),
        'momentum': (-0.5, 0.5),
        'volatility': (0, 1.5),
        'quality': (0, 2.0),
        'growth': (-0.5, 1.0)
    }

    # 初始化模块
    g.data_manager = DataQualityManager()
    g.risk_manager = RiskManager()
    g.barra_model = BarraStyleRiskModel()
    # 记录
    g.ic_history = {}
    g.performance_log = []
    set_benchmark('000300.XSHG')
    # 交易成本
    set_order_cost(OrderCost(
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5
    ), type='stock')
    run_daily(rebalance, time='09:35')
    run_monthly(calculate_ic, monthday=15)


def rebalance(context):
    """核心调仓逻辑"""
    if (context.current_dt.date() - context.run_params.start_date).days % g.rebalance_days != 0:
        return
    date = context.current_dt.date()
    log.info(f"开始调仓: {date}")
    factors_df = get_factor_data(g.stock_pool, date)
    try:
        # ===== 第一步：数据获取与质量检查 =====
        raw_factors = {}
        for name in [k for k in g.factors.keys() if k in factors_df.columns]:
            series = factors_df[name].reindex(g.stock_pool).dropna()
            series = g.data_manager.clean_data(series, method='winsorize')
            series = g.data_manager.fill_missing(
                series, 'industry_median', date, g.stock_pool)
            raw_factors[name] = series
        if not raw_factors:
            log.warning("无有效因子数据")
            return

        # ===== 第二步：因子处理与中性化 =====
        # 2.1 获取Barra风格暴露
        style_exposures = g.barra_model.calculate_style_exposures(
            g.stock_pool, date)
        # 2.2 对每个因子进行风格中性化
        processed_factors = {}
        for name, series in raw_factors.items():
            # 风格中性化
            neutral = g.barra_model.neutralize_factor(
                series, style_exposures, date=date, stock_pool=g.stock_pool, include_industry=True)
            # 标准化
            neutral = (neutral - neutral.mean()) / (neutral.std() + 1e-8)
            processed_factors[name] = neutral

        # ===== 第三步：因子合成 =====
        factor_weights = {}
        for name in processed_factors.keys():
            if name in g.ic_history and len(g.ic_history[name]) > 6:
                recent_ic = g.ic_history[name][-6:]
                avg_ic = np.mean([abs(ic) for ic in recent_ic])
                factor_weights[name] = max(0.1, avg_ic)
            else:
                factor_weights[name] = 1.0
        # 归一化权重
        total = float(sum(list(factor_weights.values())))
        factor_weights = ({k: float(v) / total for k,
                          v in factor_weights.items()})
        # 合成得分
        common_stocks = None
        for series in processed_factors.values():
            if common_stocks is None:
                common_stocks = series.index
            else:
                common_stocks = common_stocks.intersection(series.index)
        composite_score = pd.Series(0, index=list(common_stocks))
        for name, series in processed_factors.items():
            composite_score += factor_weights[name] * \
                series.reindex(common_stocks, fill_value=0)

        # ===== 第四步：选股与过滤 =====
        # 4.1 初选：因子得分前200名
        candidates = composite_score.nlargest(200).index.tolist()
        # 4.2 流动性过滤
        turnover = get_price(candidates, end_date=date,
                             count=30, fields=['money'])['money'].mean()
        candidates = [
            s for s in candidates if s in turnover.index and turnover[s] >= 5000000]
        # 4.3 基本面过滤（避免ST、退市风险）
        fundamentals = get_fundamentals(
            query(indicator.code, indicator.roe,
                  income.total_operating_revenue)
            .filter(indicator.code.in_(candidates)), date=date).set_index('code')
        candidates = [s for s in candidates
                      if s in fundamentals.index
                      and fundamentals.loc[s, 'roe'] > 0
                      and fundamentals.loc[s, 'total_operating_revenue'] > 0]

        # 4.4 行业分散
        info = get_industry(candidates, date=date)
        industry = pd.Series({
            code: info.get(code, {}).get('sw_l1', {}).get(
                'industry_name', 'Unknown')
            for code in candidates
        })
        # 按得分排序,应用行业约束
        max_per_industry = max(5, int(g.position_count * 0.15))
        final_candidates = []
        industry_counts = {}
        for code in composite_score.sort_values(ascending=False).index:
            if code not in candidates:
                continue
            ind = industry.get(code, 'Unknown')
            if industry_counts.get(ind, 0) < max_per_industry and len(final_candidates) < g.position_count:
                final_candidates.append(code)
                industry_counts[ind] = industry_counts.get(ind, 0) + 1
        # 补足数量
        if len(final_candidates) < g.position_count:
            remaining = [c for c in candidates if c not in final_candidates]
            final_candidates.extend(
                remaining[:g.position_count - len(final_candidates)])
        final_candidates = final_candidates[:g.position_count]

        # ===== 第五步：组合优化 =====
        # 获取候选股票的风格暴露
        candidate_exposures = style_exposures.loc[final_candidates]
        # 预期收益：使用标准化后的因子得分，minmax归一化
        expected_returns = composite_score.reindex(final_candidates)
        expected_returns = (expected_returns - expected_returns.min()) / \
            (expected_returns.max() - expected_returns.min() + 1e-8)
        # 优化权重

        def objective(w):
            return -np.dot(w, expected_returns)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # 风格约束
        for style, (min_exp, max_exp) in g.risk_constraints.items():
            if style in candidate_exposures.columns:
                idx = candidate_exposures.columns.get_loc(style)

                def constraint(w, i=idx, mn=min_exp, mx=max_exp):
                    exp = np.dot(w, candidate_exposures.iloc[:, i])
                    return [exp - mn, mx - exp]
                constraints.append({'type': 'ineq', 'fun': constraint})
        bounds = [(0, g.max_single_weight)
                  for _ in range(len(final_candidates))]
        x0 = np.ones(len(final_candidates)) / len(final_candidates)
        result = opt.minimize(objective, x0, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        weights = result.x if result.success else x0
        target_positions = dict(zip(final_candidates, weights))

        # ===== 第六步：风控检查 =====
        risk_check = g.risk_manager.pre_trade_check(
            target_positions,
            g.stock_pool,
            date
        )
        if not risk_check['passed']:
            log.warning(f"风控未通过: {risk_check['violations']}")
            return

        # ===== 第七步：执行交易 =====
        current_positions = set(context.portfolio.positions.keys())
        target_positions_set = set(target_positions.keys())
        for stock in current_positions - target_positions_set:
            order_target_value(stock, 0)
        for stock, weight in target_positions.items():
            px = get_current_data()[stock].last_price
            val = context.portfolio.total_value * weight * 0.98
            lot = 100
            shares = int(val / px // lot) * lot
            if shares >= lot:
                order_target_value(stock, shares * px)

        # ===== 第八步 - 组合风险监控 =====
        try:
            historical_prices = get_price(
                final_candidates,
                end_date=date,
                count=60,
                fields=['close']
            )
            if not historical_prices.empty and len(historical_prices['close']) > 20:
                returns_history = historical_prices['close'].pct_change(
                ).dropna()
                returns_matrix = returns_history[final_candidates].fillna(0)
                weights_array = np.array(
                    [target_positions[s] for s in final_candidates])
                risk_metrics = g.risk_manager.calculate_portfolio_risk(
                    weights_array,
                    returns_matrix
                )
                # 记录风险指标
                record(portfolio_var=risk_metrics['VaR_95'])
                record(portfolio_cvar=risk_metrics['CVaR_95'])
                record(portfolio_vol=risk_metrics['volatility'])
                record(portfolio_sharpe=risk_metrics['sharpe'])
                log.info(f"组合风险 - VaR: {risk_metrics['VaR_95']:.2%}, "
                         f"波动率: {risk_metrics['volatility']:.2%}, "
                         f"Sharpe: {risk_metrics['sharpe']:.2f}")
        except Exception as e:
            log.warning(f"风险计算失败: {str(e)}")

        # ===== 第九步：记录 =====
        portfolio_exposures = np.dot(weights, candidate_exposures.values)
        record(stock_count=len(final_candidates))
        record(avg_score=composite_score[final_candidates].mean())
        record(max_weight=np.max(weights))
        record(concentration=np.sum(weights**2))  # 赫芬达尔指数
        for i, style in enumerate(['size', 'value', 'momentum', 'volatility', 'quality', 'growth']):
            record(**{f'exp_{style}': portfolio_exposures[i]})
        # 保存业绩记录
        g.performance_log.append({
            'date': date,
            'stocks': final_candidates,
            'weights': weights,
            'factor_weights': factor_weights,
            'score': composite_score[final_candidates].mean()
        })
        log.info(f"调仓完成: {len(final_candidates)}只股票")

    except Exception as e:
        log.error(f"调仓失败: {str(e)}")
        import traceback
        log.error(traceback.format_exc())


def get_factor_data(stock_pool, date):
    q = query(
        valuation.code,
        valuation.pb_ratio,
        indicator.roe,
        indicator.roa,
        indicator.gross_profit_margin,
        indicator.inc_revenue_year_on_year
    ).filter(valuation.code.in_(stock_pool))
    base = get_fundamentals(q, date=date).set_index('code')
    factors_df = pd.DataFrame(index=stock_pool)
    s = base['pb_ratio'].replace([0, np.inf, -np.inf], np.nan)
    factors_df['bp_ratio'] = 1.0 / s
    factors_df['roe'] = base['roe']
    factors_df['roa'] = base['roa']
    factors_df['gross_margin'] = base['gross_profit_margin']
    factors_df['inc_revenue'] = base['inc_revenue_year_on_year']
    return factors_df

# 数据质量管理


class DataQualityManager:
    """数据质量检查与清洗"""

    def check_data_quality(self, data, data_name=""):
        """数据质量检查报告"""
        report = {
            '数据名称': data_name,
            '总样本数': len(data),
            '缺失值比例': f"{data.isna().sum().sum() / (len(data) * len(data.columns)):.2%}",
            '异常值数量': self._count_outliers(data),
            '数据时间跨度': f"{data.index.min()} 至 {data.index.max()}"
        }
        return report

    def _count_outliers(self, data):
        """统计异常值"""
        numeric_cols = data.select_dtypes(
            include=[np.number]).columns
        outliers = 0
        for col in numeric_cols:
            mean, std = data[col].mean(), data[col].std()
            outliers += ((data[col] < mean - 3*std) |
                         (data[col] > mean + 3*std)).sum()
        return outliers

    def clean_data(self, data, method='winsorize', limits=[0.01, 0.99]):
        """数据清洗：缩尾/去极值"""
        if method == 'winsorize':
            return data.clip(lower=data.quantile(limits[0]),
                             upper=data.quantile(limits[1]))
        elif method == 'mad':
            # MAD方法（中位数绝对偏差）
            median = data.median()
            mad = (data - median).abs().median()
            return data.clip(lower=median - 3*mad, upper=median + 3*mad)
        return data

    def fill_missing(self, data, method='industry_median', date=None, stock_pool=None):
        if method == 'industry_median' and date and stock_pool:
            info = get_industry(list(stock_pool), date=date)
            industry_dict = {}
            for c, v in info.items():
                industry_dict[c] = v.get('sw_l1').get('industry_name')
            industry = pd.Series(industry_dict).reindex(data.index)
            # 按行业分组，用各行业的中位数填充缺失值
            return data.groupby(industry).transform(lambda g: g.fillna(g.median()))
        return data.fillna(data.median())


# 因子研究框架
def calculate_ic(context):
    """每月计算因子IC"""
    date = context.current_dt.date()
    try:
        factors_df = get_factor_data(g.stock_pool, date)
        past_date = date - pd.Timedelta(days=20)
        prices = get_price(g.stock_pool, start_date=past_date, end_date=date, fields=['close'])
        future_returns = (prices['close'].iloc[-1] / prices['close'].iloc[0] - 1)
        for name in g.factors.keys():
            factor_values = factors_df[name].dropna()
            if factor_values.empty:
                continue
            common = factor_values.index.intersection(future_returns.index)
            ic = factor_values[common].corr(
                future_returns[common], method='spearman')
            # 更新IC历史
            if name not in g.ic_history:
                g.ic_history[name] = []
            g.ic_history[name].append(ic)
            g.ic_history[name] = g.ic_history[name][-24:]
            # 记录指标
            record(**{f'ic_{name}': ic})
            if len(g.ic_history[name]) >= 6:
                ic_series = g.ic_history[name][-6:]
                ir = np.mean(ic_series) / (np.std(ic_series) + 1e-8)
                record(**{f'ir_{name}': ir})
            log.info(f"因子{name} IC={ic:.4f}")
    except Exception as e:
        log.warning(f"IC计算失败: {str(e)}")


# 风控系统
class RiskManager:
    """风险管理系统"""

    def __init__(self):
        self.position_limits = {
            'max_single_stock': 0.15,      # 单股最大15%
            'max_industry': 0.20,          # 单行业最大20%
            'max_style_exposure': 1.0     # 风格因子暴露限制
        }

    def pre_trade_check(self, target_weights, stock_pool, date):
        """交易前风控检查"""
        violations = []
        # 1. 单股集中度
        for stock, weight in target_weights.items():
            if weight > self.position_limits['max_single_stock']:
                violations.append(f"单股{stock}权重{weight:.2%}超限")

        # 2. 行业集中度
        industry = get_industry(list(target_weights.keys()), date=date)
        industry_weights = {}
        for stock, weight in target_weights.items():
            if stock in industry:
                ind = industry.get(stock).get('sw_l1').get('industry_name')
                industry_weights[ind] = industry_weights.get(ind, 0) + weight
        # 检查行业权重限制
        for ind, weight in industry_weights.items():
            if weight > self.position_limits['max_industry']:
                violations.append(f"行业{ind}权重{weight:.2%}超限")

        # 3. 流动性检查
        illiquid = self._check_liquidity(list(target_weights.keys()), date)
        if illiquid:
            violations.append(f"流动性不足股票: {illiquid[:5]}")
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }

    def _check_liquidity(self, stocks, date, min_turnover=5000000):
        """流动性检查：日均成交额>500万"""
        turnover = get_price(stocks, end_date=date, count=20,
                             fields=['money'])['money'].mean()
        return [s for s in stocks if s in turnover.index and turnover[s] < min_turnover]

    def calculate_portfolio_risk(self, weights, returns_history):
        """组合风险计算"""
        if len(returns_history) < 20:
            return {'VaR_95': 0, 'CVaR_95': 0, 'volatility': 0}
        # 历史模拟法计算VaR
        portfolio_returns = returns_history @ weights
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        vol = portfolio_returns.std() * np.sqrt(252)
        return {
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'volatility': vol,
            'sharpe': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        }


# 因子中性化与风险模型（Barra风格）
class BarraStyleRiskModel:
    """Barra风格风险模型"""

    def __init__(self):
        self.style_factors = ['size', 'value',
                              'momentum', 'volatility', 'quality', 'growth']

    def calculate_style_exposures(self, stock_pool, date):
        """计算风格因子暴露"""
        exposures = pd.DataFrame(index=stock_pool)
        # Size: 市值对数
        try:
            market_cap = get_fundamentals(
                query(valuation.code, valuation.market_cap)
                .filter(valuation.code.in_(stock_pool)), date=date
            ).set_index('code')['market_cap']
            exposures['size'] = self._standardize(np.log(market_cap))
        except:
            exposures['size'] = 0

        # Value: EP(市盈率倒数)
        try:
            pe = get_fundamentals(
                query(valuation.code, valuation.pe_ratio)
                .filter(valuation.code.in_(stock_pool)), date=date
            ).set_index('code')['pe_ratio']
            exposures['value'] = self._standardize(1 / pe.replace(0, np.nan))
        except:
            exposures['value'] = 0

        # Momentum: 过去12个月收益
        try:
            prices = get_price(stock_pool, end_date=date,
                               count=252, fields=['close'])
            mom = (prices['close'].iloc[-1] / prices['close'].iloc[0] - 1)
            exposures['momentum'] = self._standardize(mom)
        except:
            exposures['momentum'] = 0

        # Volatility: 波动率（负值=低波）
        try:
            prices = get_price(stock_pool, end_date=date,
                               count=60, fields=['close'])
            vol = prices['close'].pct_change().std()
            exposures['volatility'] = -self._standardize(vol)
        except:
            exposures['volatility'] = 0

        # Quality: 毛利率
        try:
            gross_margin = get_fundamentals(
                query(indicator.code, indicator.gross_profit_margin)
                .filter(indicator.code.in_(stock_pool)), date=date
            ).set_index('code')['gross_profit_margin']
            exposures['quality'] = self._standardize(gross_margin)
        except:
            exposures['quality'] = 0

        # Growth: 用净利润增速
        try:
            profit_growth = get_fundamentals(
                query(indicator.code, indicator.inc_net_profit_year_on_year)
                .filter(indicator.code.in_(stock_pool)), date=date
            ).set_index('code')['inc_net_profit_year_on_year']
            exposures['growth'] = self._standardize(profit_growth)
        except:
            exposures['growth'] = 0

        return exposures.fillna(0)

    def _standardize(self, series):
        """标准化"""
        return (series - series.mean()) / (series.std() + 1e-8)

    def neutralize_factor(self, factor_data, style_exposures, date=None, stock_pool=None, include_industry=True):
        """对因子进行中性化：默认 风格 + 行业（可选）factor_data: Series(index=code),style_exposures: DataFrame(index=code, cols=[size,value,...])"""
        common = factor_data.index.intersection(style_exposures.index)
        y = factor_data.loc[common].astype(float)
        X = style_exposures.loc[common].astype(float)

        # 添加行业哑变量
        if include_industry and (date is not None):
            ind_dum = self._make_industry_dummies(list(common), date)
            ind_dum = ind_dum.reindex(index=common).fillna(0)
            X = pd.concat([X, ind_dum], axis=1)

        X = X.fillna(0)
        y = y.fillna(y.median())
        reg = LinearRegression()
        reg.fit(X.values, y.values)
        residual = y - reg.predict(X.values)
        return pd.Series(residual, index=common)

    def _make_industry_dummies(self, stock_pool, date):
        info = get_industry(list(stock_pool), date=date)
        ind = pd.Series({c: info.get(c).get('sw_l1').get(
            'industry_name') for c in stock_pool})
        dummies = pd.get_dummies(ind.astype(str), drop_first=True).reindex(
            list(stock_pool), fill_value=0)
        return dummies
