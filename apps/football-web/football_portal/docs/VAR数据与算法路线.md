# Django、VAR辅助复核与数据路线（2026-10-26）

本轮已新增真实数据下载与42项目录，见`真实数据下载说明.md`和`42类事件与算法接入.md`。下文10个原研究来源与10条官方案例的访问状态分别保留；本轮实际取得的是另外的Infactory研究子集与StatsBomb结构化事件，未把受限SoccerNet录像标成已下载。运行监控展示本地实际文件，VAR选择器已扩展为共享42项目录；自动事件模型仍未训练。

## 页面确实由Django运行

`manage_football.py`启动Django；`project_settings.py`配置应用、模板和独立SQLite；`project_urls.py`分发URL；`views.shell`调用Django的`render()`返回HTML模板。任务、助手和复核案例分别通过Django ORM模型保存。页面上的CSS、JavaScript和SVG负责交互与样式，不改变Django作为Web框架的事实。

新增VAR相关后端：`var_models.py`、`var_views.py`、`var_services.py`、`var_urls.py`及`0002_var_review_cases.py`迁移。数据库仍是新增模块的`.local/portal.sqlite3`；原工程文件和原数据库不改。

## 项目应升级成什么

建议项目定位为“足球赛事事件识别与VAR辅助复核系统”。原四类YOLO只提供目标位置，事件识别还需要时序、身份、场地几何和规则证据。当前已经实现Web侧人工复核工作台，尚未接入自动越位、犯规、门线和出牌事件模型。不能把手绘线、浏览器时间步进或LLM解释称作正式VAR技术。

| 任务 | 需要的数据/算法 | 现阶段接口与页面 |
| --- | --- | --- |
| 越位 | 传/触球时刻、球和球员轨迹、球队身份、可计入身体部位、倒数第二名对手、场地和相机标定 | 越位主题、慢放、关键帧、未校准参考线、人工意见；自动判罚待接入 |
| 冲撞/犯规 | 接触前后片段，多视角，动作类型与严重程度标注 | 多个关键帧和观察记录；不能由有人倒地直接断言犯规 |
| 进球/门线 | 整个球体过线真值、球门/球门线几何、可靠视角；遮挡时需要补充视角或专用技术 | 进球主题和证据不足记录；不以球心越过一条图像线判进球 |
| 裁判出牌/球员受牌 | 小目标红黄牌检测、裁判手/持牌动作、时间线、受牌者身份与历史黄牌 | 出牌主题与证据记录；牌色、受牌人和是否应该出牌是不同标签 |

## 已找到的近期官方图片与视频入口

以下是10条官方素材线索，不是已经下载、标注好的训练集。8条为2026近期案例，另2条是明确标为历史的门线案例。公开文章可读不等于视频可播放或已获训练许可；X嵌入媒体部分403，FIFA+部分跳转DAZN。

| 案例 | 比赛日期 / 发布或更新日 | 事件与定位 | 来源 |
| --- | --- | --- |
| 水晶宫—纽卡：越位进球取消 | 2026-10-25 / 2026-10-25 | 越位、进球/门线；Matchweek 8 Sunday，52分钟，Joe Willock | [官方入口](https://www.premierleague.com/en/news/4744845/matchweek-8-var-decisions-explained) |
| Hull—Brentford：禁区接触与黄牌 | 2026-10-25 / 2026-10-25 | 冲撞/犯规、出牌/受牌；Matchweek 8 Sunday，21分钟，Hickey与Iroegbunam | [官方入口](https://www.premierleague.com/en/news/4744845/matchweek-8-var-decisions-explained) |
| Hull—Brentford：接触但不判点球 | 2026-10-25 / 2026-10-25 | 冲撞/犯规；Matchweek 8 Sunday，61分钟，Sangare与Belloumi | [官方入口](https://www.premierleague.com/en/news/4744845/matchweek-8-var-decisions-explained) |
| Ipswich—Nottingham Forest：滑铲复核后黄牌 | 2026-10-23 / 2026-10-25 | 冲撞/犯规、出牌/受牌；Matchweek 8 Friday，48分钟，Issa Diop | [官方入口](https://www.premierleague.com/en/news/4744845/matchweek-8-var-decisions-explained) |
| Aston Villa—Viking：在位，进球有效 | 2026-10-21 / 2026-10-22 | 越位、进球/门线；Matchday 3 Aston Villa vs Viking，87分钟 | [官方入口](https://www.uefa.com/uefachampionsleague/news/02a9-218080c33e00-948c7f405b79-1000--uefa-champions-league-var-technical-explanations/) |
| Atlético—Manchester United：DOGSO红牌 | 2026-10-13 / 2026-10-22 | 冲撞/犯规、出牌/受牌；Matchday 2 Atleti vs Manchester United，75分钟 | [官方入口](https://www.uefa.com/uefachampionsleague/news/02a9-218080c33e00-948c7f405b79-1000--uefa-champions-league-var-technical-explanations/) |
| Brentford—Liverpool：头顶冲撞后红牌 | 2026-10-17 / 2026-10-18 | 冲撞/犯规、出牌/受牌；Matchweek 7 Saturday，45+8分钟，Yarmoliuk与Jacquet | [官方入口](https://www.premierleague.com/en/news/4737981) |
| Manchester United—Manchester City：Foden被出示红牌 | 2026-09-13 / 2026-09-14 | 冲撞/犯规、出牌/受牌；Matchweek 4 Sunday，23分钟；题图Haaland celebrates while Foden is shown red | [官方入口](https://www.premierleague.com/en/news/4713977/matchweek-4-var-decisions-explained) |
| Sweden—USA女足：历史门线技术案例 | 2023-08-06 / 2023-08-06 | 进球/门线；点球大战Lina Hurtig罚球；正文GLT段落与旁边图片 | [官方入口](https://www.svenskfotboll.se/nyheter/landslag/2023/08/vm-swe-usa/) |
| France—Honduras：历史GLT进球确认 | 2014-06-15 / 2018-04-27 | 进球/门线；48分钟，Benzema射门反弹、Valladares碰球后的门线判定 | [官方入口](https://inside.fifa.com/tournaments/mens/worldcup/2018russia/news/48-days-to-go-was-it-over-the-line-2945114) |

### 访问与授权记录

- **水晶宫—纽卡：越位进球取消**：官方图文页及嵌入判罚视频帖。正文可读；官方X视频帖本次返回403，未核验播放、未下载。未见开放训练许可；公开观看不等于训练或再分发授权。
- **Hull—Brentford：禁区接触与黄牌**：官方图文页及嵌入判罚帖。正文可读；X帖本次403，未取得录像。训练、截帧及公开再分发权利需另外确认。
- **Hull—Brentford：接触但不判点球**：官方图文页及判罚解释。正文可读；嵌入媒体播放未核。没有据此取得原媒体训练授权。
- **Ipswich—Nottingham Forest：滑铲复核后黄牌**：官方犯规图片、图文及嵌入判罚帖。正文和题图链接可定位；图片抓取超时、X帖403，未下载。未见开放训练许可；需向权利方确认截帧和使用范围。
- **Aston Villa—Viking：在位，进球有效**：官方VAR技术解释页；页内附部分图片或视频。正文可读，动态嵌入媒体未取得原文件。UEFA页声明版权；未获得开放训练或再分发许可。
- **Atlético—Manchester United：DOGSO红牌**：官方说明和对应判罚媒体区。正文可读；页面说明重大判罚配图或视频，未取得独立录像。未见可直接训练的开放许可。
- **Brentford—Liverpool：头顶冲撞后红牌**：官方图文页及嵌入复核判罚帖。正文可读；X帖403，录像未下载。公开链接不授予训练和重新发布录像的权利。
- **Manchester United—Manchester City：Foden被出示红牌**：官方发牌主题图片及判罚说明。正文和题图链接已核到；未保存原图或视频。图片为官方媒体资产，训练与公开复用需另外确认。
- **Sweden—USA女足：历史门线技术案例**：足协官方图片/报道；另有FIFA+集锦入口。足协正文可读；FIFA+集锦现在重定向DAZN，账号及地区限制需用户核验。足协注明图片通常归Bildbyran；未取得原图或录像训练许可。
- **France—Honduras：历史GLT进球确认**：FIFA官方回顾；另有FIFA官方GLT专门视频入口。FIFA回顾正文可读；YouTube视频页本次抓取失败，播放未核验。未发现开放训练许可；不能将公开回顾当作原始标定数据。

## 训练与研究数据

先准备SoccerNet-MVFouls与17类Action Spotting；几何与身份数据作为后续补充。最新比赛素材适合另建外部验证案例，但必须先取得允许使用的录像并重新标注。不能用官方文章中的事件时间戳代替精确触球帧或3D几何真值。

### SoccerNet-MVFouls：多视角犯规与判牌参考

- [项目主来源](https://www.soccer-net.org/tasks/multi-view-foul-recognition)。
- 规模与格式：作者README列3901个动作：2916训练、411验证、301测试、273隐藏挑战；每个动作至少现场与一个回放视角；HF约21.3GB。
- 标注：犯规/无犯规与无牌/黄牌/红牌4级；8类动作：站立铲抢、铲抢、拉扯、推人、争抢、假摔、高抬腿、肘击；10种动作属性，视频片段与JSON。
- 获取：HF公开包；内部转播片段仍需SoccerNet NDA及解压密码。访问流程以官方最新Data页为准，未代签或申请。
- 许可：HF卡列GPL-3.0；视频仍有NDA条件，不能以代码许可证替代录像授权。
- 用法与局限：优先级最高，适合冲撞/犯规复核与判牌建议。参考论文VARS/CVPR2023，https://arxiv.org/abs/2304.04617 。原挑战主要2014–2017赛季，不是2026最新比赛；应显示模型建议与实际裁判决定两个字段。
- 本项目状态：未下载。
- [下载/申请入口](https://huggingface.co/datasets/SoccerNet/SN-MVFouls-2025)。

### SoccerNet Action Spotting：17类事件时间线

- [项目主来源](https://www.soccer-net.org/tasks/action-spotting)。
- 规模与格式：500场完整转播、另50场challenge；25fps的720p/224p视频；部分任务有2fps预提取特征。
- 标注：17类单时间戳事件；进球、越位、犯规、黄牌、红牌、两黄变红等；Labels-v2.json；原视频为上下半场mkv。
- 获取：标签库SN-Labels公开；原视频SoccerNet_raw_HQ需HF账号、申请与NDA审批。旧Google NDA表已弃用，见https://www.soccer-net.org/data。
- 许可：公共标签与受保护视频的访问范围不同；视频按SoccerNet NDA使用，本次未核验可商用授权。
- 用法与局限：事件检索和复核入口的第一优先。Goal/Offside单时间戳不能当作门线3D真值、精确越位线或出脚帧真值。官方包说明https://github.com/SoccerNet/SoccerNet 明确SN-Labels public与原录像gated。
- 本项目状态：未下载。
- [下载/申请入口](https://huggingface.co/datasets/SoccerNet/SN-Labels)。
- [公开标签示例直链](https://huggingface.co/datasets/SoccerNet/SN-Labels/resolve/main/england_epl/2014-2015/2015-02-21%20-%2018-00%20Chelsea%201%20-%201%20Burnley/Labels-v2.json)。
- 示例核验：HF官方文件树已核验此文件存在，37609字节；本机resolve/raw正文请求遇到SSL EOF，未成功读取或保存正文。

### SoccerNet Calibration / Field Localization：场地与球门几何

- [项目主来源](https://www.soccer-net.org/tasks/camera-calibration)。
- 规模与格式：20,028张事件及回放图片，另2104张challenge；HF约3.98GB。
- 标注：球场线条/圆弧/球门部件；相机参数及重投影评估任务；图片与几何标注。
- 获取：官方HF文件页存在，viewer当前解析报错；具体文件访问和协议从SoccerNet官方Data页核验。
- 许可：当前HF卡未给出完整许可说明，不能承诺商用或免协议。
- 用法与局限：仅为VAR几何辅助基础，不是越位或门线判罚数据。还需要出脚时刻、可用于越位的身体部位、球队、倒数第二防守队员；单目透视/遮挡不足时应输出证据不足。
- 本项目状态：未下载。
- [下载/申请入口](https://huggingface.co/datasets/SoccerNet/SN-Calibration-2023)。

### SoccerNet-GSR：球员身份、球队和场地位置

- [项目主来源](https://www.soccer-net.org/tasks/game-state-reconstruction)。
- 规模与格式：论文/代码总200段30秒序列；公开划分57训练、59验证、50测试；1080p主镜头；HF约35.1GB。
- 标注：人员位置/轨迹、角色、球队和球衣号码；场地线条和几何；Labels-GameState.json，官方建议版本至少1.3。
- 获取：官方HF有公开图像预览和下载入口；本次未实际下载。
- 许可：HF卡标GPL-3.0，具体转播素材使用条件仍应与官方核对。
- 用法与局限：适合参与者跟踪、身份归属及小地图，辅助回答谁参与冲撞/谁受牌；本身不提供出牌动作、判罚或越位真值。勿把200总段数混成200个公开有标注训练样本。论文https://arxiv.org/abs/2404.11335。
- 本项目状态：未下载。
- [下载/申请入口](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025)。

### SoccerNet Ball Action Spotting：射门、抢断和进球片段

- [项目主来源](https://www.soccer-net.org/tasks/ball-action-spotting)。
- 规模与格式：2024基准7场720p英格兰联赛+2场challenge；2025为Team Ball Action Spotting，HF约19.3GB。
- 标注：12类球相关动作单时间戳；传球、射门、封堵、成功抢断、任意球、进球等；2025加入球队归属任务。
- 获取：HF gated；需账号、提交申请、SoccerNet NDA与人工审批，批准后ZIP无需密码。
- 许可：HF标other，以SoccerNet NDA为准。
- 用法与局限：适合射门过程与候选片段时间线；不是精确球3D轨迹、整球过线标签或犯规成立依据。2024的规模不能直接当成2025新增规模。
- 本项目状态：未下载。
- [下载/申请入口](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025)。

### SoccerReplay-1988：比赛事件与解说

- [项目主来源](https://huggingface.co/datasets/Homie0609/SoccerReplay-1988)。
- 规模与格式：1988场，训练/验证/测试为1488/250/250；每场上下半场mkv与事件、解说和比赛背景JSON。
- 标注：时间对齐事件描述；自然语言解说和比赛背景；由作者自动整理管线生成的数据，非精细几何判罚真值。
- 获取：需HF登录、NDA申请；文本文件批准后可访问，完整视频存储链接另外发放，作者提示审核可能数周。
- 许可：CC BY-NC-SA 4.0，作者明确非商用与NDA要求。
- 用法与局限：正确名称为1988，不是198；CVPR2025论文https://openaccess.thecvf.com/content/CVPR2025/papers/Rao_Towards_Universal_Soccer_Video_Understanding_CVPR_2025_paper.pdf 。适合LLM讲解与检索，不是立即可用的2026比赛或精确判罚标注。
- 本项目状态：未下载。
- [下载/申请入口](https://huggingface.co/datasets/Homie0609/SoccerReplay-1988)。

### IAUFD：含红黄牌与裁判的足球图片

- [项目主来源](https://sites.google.com/view/image-and-video-analysis/iaufd)。
- 规模与格式：作者页列100,000张图片、33段源视频、10种图像事件类别。
- 标注：红牌、黄牌、裁判、球、球门、罚球区等图像事件类；txt/xlsx标签；分卷与小型数据版本。
- 获取：作者主页给出Google Drive分卷、标签及小型版入口；本次未逐个验证文件访问。
- 许可：作者主页未明确列出，使用前联系作者确认。
- 用法与局限：可作为出牌/球门图片筛选线索，不是被出牌球员身份或判罚全过程标注；不能把连续帧随机拆分到训练/测试，也不能把页面公开理解为任意再分发。
- 本项目状态：未下载。
- [下载/申请入口](https://sites.google.com/view/image-and-video-analysis/iaufd)。

### Card_Detection：红黄牌小目标检测起点

- [项目主来源](https://universe.roboflow.com/neural-network-ep0gh/card_detection-fuysv)。
- 规模与格式：网页当前164原图，版本1为423张增强图；作者模型卡另称约141独立来源，信息有差异。
- 标注：网页4类：红牌、黄牌、绿牌、objects杂项；作者模型卡称3类，与导出应重新核对；目标检测框，可按平台提供格式导出。
- 获取：Roboflow Universe公开项目，导出通常需平台账号；版本页本次请求超时，未实际导出。作者模型卡https://huggingface.co/rarfileexe/Football-Referee-Card-Detector。
- 许可：项目页列CC BY 4.0；作者模型权重另列AGPL-3.0，二者应分别记录。
- 用法与局限：仅为小规模起点，需查data.yaml、清除杂项与误标、合并重复/增强图后按来源分组。作者自述无独立测试集；不能把检测牌等同于识别受牌球员或判断是否应给牌，不直接采用绿色牌作为本项目标准判罚类别。
- 本项目状态：未下载。
- [下载/申请入口](https://universe.roboflow.com/neural-network-ep0gh/card_detection-fuysv/dataset/1)。

### 裁判手势YOLOv8s研究：出牌与越位示意

- [项目主来源](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1341234/full)。
- 规模与格式：2024论文自建1200张、6种裁判手势；源比赛2018–2023。
- 标注：整名裁判的检测框与手势类型；主裁出牌、任意球/点球等；助理裁判越位示意等。
- 获取：论文可读；本次未找到明确可直接下载的完整数据ZIP。文章数据可用声明指文章/补充材料与联系作者，需进一步确认。
- 许可：论文开放许可不等于独立数据包许可；数据授权待核验。
- 用法与局限：可学习YOLO小目标/手势设计，不能当作已经取得的训练集。识别裁判做出的手势只说明观察到裁判信号，不证明原犯规或越位判断正确。
- 本项目状态：未下载。

### 静态越位位置数据与方法参考（Panse 2020）

- [项目主来源](https://github.com/Neerajj9/Computer-Vision-based-Offside-Detection-in-Soccer)。
- 规模与格式：作者仓库公开数据Drive入口与示意图；样本总量、独立数据许可及Drive完整可用性未验证。
- 标注：静态场景的越位位置方法参考；姿态、球队颜色、场地消失点相关代码。
- 获取：作者GitHub公开；Drive数据入口存在，本次未下载或代申请。
- 许可：代码MIT；不能自动把MIT套用于转播截图或其他第三方数据。
- 用法与局限：论文DOI https://doi.org/10.1145/3422844.3423055 。适合静态辅助线研究，不是最新比赛数据，单张图的越位位置判断不等于越位犯规成立。
- 本项目状态：未下载。
- [下载/申请入口](https://drive.google.com/drive/folders/1TgxT-9GRB3BWice_5WHuCQ4Byev-NHFI?usp=sharing)。

## 取得数据后怎样整理

1. 登记来源、比赛日期、媒体URL、许可、原始文件SHA256、视角、帧率、时长与标注者。原始数据只放`.local/`，不提交Git。
2. 用比赛ID和事件ID分组；同一事件的多视角、回放、连续帧和增强图不能跨训练/验证/测试集合。
3. 为每个事件保留发生前后的上下文，单独标注触球/接触/举牌/受牌等关键时刻。保留无犯规、非越位和未进球的对照负例。
4. 至少双人复核争议标签，分开保存“视频中观察到的动作”“实际裁判决定”“模型候选”“人工复核意见”。证据不足可标未知，不强行二分类。
5. 时间线用事先固定的容差评估Precision/Recall/F1；犯规类别与严重程度用逐类/宏平均F1；牌检测AP和受牌者关联分别评估。几何误差需要专门真值，不以四类框mAP代替判罚性能。

## 与算法同学衔接

你负责Django复核台、事件时间线、证据与LLM讲解；同学负责数据标注和事件模型。原四类权重仍可用于定位，新增牌/事件模型应走独立适配层，不能强塞进当前只接受四类的旧适配器。

建议事件候选契约如下，**这是待实现的模型接口约定，当前不提供自动事件导入或自动判罚API**：

```json
{
  "schema": "football.event-candidate.v1",
  "source": "model_candidate",
  "event_type": "foul",
  "start_time": 12.4,
  "end_time": 15.0,
  "score": 0.73,
  "model_id": "provided-by-algorithm-owner",
  "views": [
    "broadcast",
    "replay"
  ],
  "actor_ids": [],
  "decision_status": "pending_human_review",
  "limitations": [
    "example values only; not experimental results"
  ]
}
```

LLM后续读取已核验的事件、关键帧说明、规则版本与证据缺口，生成解释，不擅自更改判罚字段。当前LLM模块仍只解释已接入的检测统计，VAR导出为本地人工复核报告；未声称已调用视觉大模型或已训练事件模型。

## 本次可以直接使用的Django VAR功能

独立上传图片/视频或复用已有素材，不需要先运行YOLO；视频慢放0.25×/0.5×、浏览器40ms时间步进、保存原始关键帧、单独存储归一化参考线、人工观察/意见、需要关键帧和意见后才能标为已人工复核，下载JSON/Markdown证据报告。原视频采用会话私有接口和Range流式读取。

API根路径为`/football/api/var/`：`cases/`、`cases/<id>/`、`cases/<id>/keyframes/`、`cases/<id>/keyframes/<frame_id>/`、`cases/<id>/artifact/<key>/`、`cases/<id>/report.json`、`cases/<id>/report.md`；公开数据索引为`resources/`。修改接口需要CSRF，案例和素材按浏览器会话隔离。

## 规则版本与关键依据

- [IFAB Law 11：越位](https://www.theifab.com/laws/latest/offside/)：区分越位位置与越位犯规，核对触球时刻、球和倒数第二名对手；手臂不计入。 公开规则页面；当前2026/27，具体比赛按所用版本。
- [IFAB Law 10：进球与门线](https://www.theifab.com/laws/latest/determining-the-outcome-of-a-match/)：需要整个球体越过门线并位于门柱之间、横梁之下；还需排除进球前犯规。 公开规则页面；不是已取得门线传感器或三维真值数据。
- [IFAB Law 12：犯规与不正当行为](https://www.theifab.com/laws/latest/fouls-and-misconduct/)：碰撞不自动等于犯规；完整动作与不同视角有助于区分不慎、鲁莽和过分力量。 公开规则页面；不由单张图的人员框直接生成判罚。
- [IFAB 2026/27 VAR协议](https://www.theifab.com/laws/latest/video-assistant-referee-var-protocol/)：普通黄牌记录不自动属于VAR介入；当前协议包括明确错误的第二黄牌导致红牌的复核，但不包括漏掉潜在第二黄牌；角球纠错是赛事可选规则。 公开协议，核验2026-10-26。最终以相应比赛规程和主裁决定为准。

规则核验日为2026-10-26，页面显示2026/27。正式比赛仍以该比赛采用的规则版本为准。门线旧案例没有冒充2026原始标定数据；尚未取得可直接验证毫米级越位或门线判断的最新多机位真值。

## 当前验证范围

新增VAR后端15项与监控/配置8项检查通过；原有Django/绘制34项一并通过，共57项。真实浏览器已验证独立视频上传、0.25×慢放、参考线、原始截图与线条分离、人工记录保存、Range206、会话404、CSRF403及5种宽度。验证短片由重复图片构成，只证明接口工作，不代表真实比赛判罚效果。
