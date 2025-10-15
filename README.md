#Quant-Factor-Framework  
一个基于聚宽平台的完整量化多因子选股与风险控制框架

#项目简介
本项目实现了一个系统性的量化研究与实盘策略框架，涵盖：
- 因子数据管理与质量检测 (`DataQualityManager`)
- 风控系统与行业集中度限制 (`RiskManager`)
- Barra 风格风险模型中性化 (`BarraStyleRiskModel`)
本项目可直接在聚宽研究环境/回测环境运行，


#框架结构
main.py
├── DataQualityManager         # 数据质量与异常值检测
├── RiskManager                # 单股、行业、风格、流动性风险控制
├── BarraStyleRiskModel        # 风格暴露计算与因子中性化
└── initialize() / rebalance() # 核心调仓逻辑

#快速开始
# 1. 克隆仓库
git clone https://github.com/0xooank/quent-factor-framework.git
cd quent-factor-framework

# 2. 安装依赖
pip install -r requirements.txt

# 3. 在聚宽平台运行 main.py
