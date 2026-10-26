# 足球Web与LLM扩展

完整保留已实现的Django页面、VAR人工复核、运行监控、检测结果与LLM适配，以及42项目录和数据下载工具；不改ODPlatform原代码。

这是同学[defect_manage](https://github.com/chenkangkang999/defect_manage)的扩展源码，**需要该Django宿主，不能在本目录直接独立启动**。原`apps/web-backend`保持不变。

使用方法：

1. 将本目录的`football_portal/`和`manage_football.py`复制到`defect_manage`根目录；目标有同名文件时先比较，不覆盖现有修改。
2. 在宿主根目录执行`football_portal/scripts/setup.ps1`，复制模块`.env.example`为`.env`。
3. 配置`FOOTBALL_PLATFORM_ROOT`指向ODPlatform，并填写同学确认的权重及算法解释器；执行`football_portal/scripts/start.ps1`。

打开`http://127.0.0.1:8502/football/`。Web使用独立Python3.12+环境及数据库。

详细[安装与运行](football_portal/docs/安装与运行.md)、[仓库交付说明](REPOSITORY_NOTES.md)、[数据下载](football_portal/docs/真实数据下载说明.md)、[配置](football_portal/docs/配置与扩展说明.md)均在模块内。

只上传源码、配置模板、测试、说明和生成的装饰背景；本地200个数据文件、权重、密钥、环境和数据库不上传。42项为目标清单，自动事件模型未训练，当前VAR为人工复核。
