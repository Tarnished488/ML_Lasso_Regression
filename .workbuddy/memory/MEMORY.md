# Week3 ML_Lasso_Regression 项目笔记

## 项目约定（用户明确要求）

- **代码文件全英文**：注释、docstring、print 输出、异常消息一律英文，禁止中文（2026-10-11 用户明确要求）。作业报告正文仍可用中文，但**提交的代码必须是纯 ASCII**。用 `grep -nP '[^\x00-\x7F]'` 自查。
- **代码风格：ponytail 懒人模式**（用户主动启用）。最小改动、不加没人要的抽象、docstring 精简。改代码前先导出 `inspect.signature` 基线到 /tmp，改完 diff 比对，确保签名零变化。
- 结构：`Src/Data_loader.py`（数据加载+全部绘图函数）/ 每模型一个 .py / `Figures/` 输出目录。

## 运行环境

- 用**系统 Python 3.14**：`C:/Users/yqy08/AppData/Local/Programs/Python/Python314/python.exe`（numpy 2.4.3 / pandas 3.0.1 / matplotlib 3.10.8 / sklearn 1.8.0 都有）。
- 托管 Python 3.13 **没装 numpy**，别用它跑本项目。
- 跑图时加 `MPLBACKEND=Agg` 避免 plt.show() 阻塞。

## 数据集

- `week3.php.csv`：首行 `# id:19-19-19`（注释，即数据集标识，**要粘贴进提交报告**），后 199 行 `x1,x2,y`。
- **x2 与 y 强相关 r≈0.880；x1 几乎无关 r≈0.061** —— 这就是 Lasso 该把 x1 系数压成 0 的关键现象。

## 作业要求要点（TCD CS7CS4 / CSU44061 Week3）

- (i)(a) 3D 散点；(b) 多项式特征(power 5) + Lasso 扫 C，找"系数全 0"的小 C 起点；(c) 网格预测 + 3D 曲面图，网格要略超数据范围但不能太大；(d) 解释欠/过拟合；(e) 换 Ridge 重做并对比。
- (ii)(a) 5-fold CV 误差 errorbar vs C；(b) 选 C 并解释；(c) Ridge 重做。
- **sklearn `Lasso(alpha=...)`；作业的 C 与 alpha 关系：alpha = 1/(2C)**。小 C = 大惩罚 = 系数被压 0。
- 作业硬性绘图要求：轴必须有标签、所有文字要足够清晰可读、正文必须解释每张图。（已用全局 rcParams 放大字号满足）

## Data_loader.py 当前状态（2026-10-11）

- 235 行，纯 ASCII，公开函数签名保持稳定（有 /tmp 基线可比对）。
- 可用函数：`load_data` / `load_dataframe` / `summarize_data` / `make_grid` / `plot_data_3d`(i a) / `plot_fit_surface`(i c,e) / `plot_cv_errorbar`(ii a) / `plot_coef_vs_C`(i b,d,e) / `save_figure` / `new_figure`。
- 已生成 `Figures/data_3d.png`（作业 (i)(a)）。
- `__main__` 自检块用假数据冒烟测试全部绘图函数，直接 `python Data_loader.py` 可全量验证。
