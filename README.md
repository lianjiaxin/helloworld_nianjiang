重点问题及建议汇总
系统开发：
	Tacos系统，目前主要是用于组内工具发布，能否实现需求提交、缺陷提交、文档评审、缺陷统计等管理工具开发的各个生命周期。当前issue可以管理，但是无法统计数据，领导想看工作量，类似ET数据，Tacos能否实现？
	由于技术开发组更多是在平台角度考虑，所以很多时候没有考虑到产品使用的一些便捷性或者使用要求，一些平台搭建的时候，最好能预先收集下产品这边的需求
	想兼容各个产品组，兼容各个功能，但有时候做的产品可能大而全，会导致上线会慢，为了兼容会导致各个产品组用起来都不那么顺畅
	把类似提问解答的平台或者说是部分氛围构建起来，让大家的问题能得到快速解答

HITA框架
	HITA本身的问题，HITA本身执行的时候发现由于脚本语言，运行速度较慢，此外有些脚本每次执行反馈的结果会不一样（比方有时候可以通过有时候无法通过），也没有发现问题，HITA整体并发处理能力较弱，长时间稳定性较弱，等等想看下是否有这些优化这些已有问题的计划。以上问题导致我们非用例的自动化已经慢慢转为用python了。
	HITA中，新接口的关键字添加不及时 （朱小龙7、王梦茹）
	HITA安装路径改在C盘后，占用很多存储空间，目录下还有一些以日期结尾的备份hita, 工作机C盘本身只有100G会造成存储不足  （HITA的备份文件的大小为1个G，确实太大了）
	部分海外纯英文操作系统无法离线安装HITA包，需要手动解压，手动修改环境变量等。 （英文版没详细考虑，兼容性存在问题。后续专门出英文版本）
	HITA调用接口的速度慢，同样一个接口，使用HITA要10s才会返回结果，使用Jmeter 1s内返回结果
建议：优化HITA处理接口的速度
	针对公司内通用的的几种加密方式进行统计生成关键字，在使用HITA时，对于用户登录/新增时的密码加密问题，这也是我们目前使用工具进行接口测试时的一个难点
	关键字执行效率低，尤其以cs，bs为重，cs关键字单执行耗时平均3s，bs关键字必须每次都要新打开浏览器，调试效率很低；
	RIDE框架不适合编写代码逻辑稍微复杂的用例，效率比利用python直接开发更低
	经过开发实践，个人觉得 用例-步骤-操作-变量，可以简化为 用例-关键字-变量 （确认下TOD默认的模板结构，这个结构应该是涛涛那边早期用的脚本生成的架构，偏向骆军强的UI类脚本开发规范，不见得就是最佳的）
过程改进
	针对于我们对于各个平台的需求，以及这些需求的规划，好像没有成为一个闭环，我们不知道哪些需求是被接受了，会在哪个版本实现等等信息，你们发布的各个版本具体增删改了什么东西，我们也不是很清楚。
	是否考虑在测试部范围内，成立专门的自动化测试和维护小组，负责可自动化的用例测试，并负责自动化用例维护，接入不同小组的自动化测试，将手工测试与自动化测试分离开
	跟随产品组的自动化支撑感觉没有流程。产品组现在其实有很多自动化的需求，但不知道往谁那边提，怎么支持，感觉整体的接口不是很清晰，大部分就是靠口头去找，去沟通
	之前的自动化主要围绕反复操作、压力及少部分的功能，2019也希望有所突破，也希望和技术开发组更好地对接解决业务组的问题。2019年如何提效是传输测试这边的重中之重。
	如何将项目业务和接口测试结合，目前只进行接口正向测试，测试意义不是太大；希望能够一起讨论一下，提供一下接口测试的方向
	之前做的都是UI自动化，客户端界面变化了，之前写的脚本就要进行维护（成本较大），导致我们后面较少的使用自动化测试；是否有哪些改进方案能够优化
内部其他系统
	T-lab运行虚拟机环境不够稳定，之前用了几次，创建经常失败；其他操作用户体验也不是很好（操作异常无明确反馈）；
	测试资源的利用，目前测试部的资源使用主要实现借用信息登记；而资源是否得到合理利用，各组资源分布情况，资源的线下借用流程简化，查询便捷等还有改进空间；（之前有同事反馈可以做一下库房管理）
	数据展示，可结合ET项目登记情况展示某时间段内的甘特图和任务量，便于各组评估分配人力投入；（同事反应研发也希望根据测试部的测试安排及时调整提测情况）
	任务管理，组内对任务这块的实现情况需求还比较大，了解到已有相应计划去优化任务管理功能；希望后续能积极应用起来；
	缺陷提交，可考虑提供给测试人员更便捷的服务，如问题截图，操作录制，缺陷信息获取，沟通记录保存等；


自动化需求
	ehome协议测试
	是否能协助产品组进行一些技术上的突破，产品组测试这边整体的自动化比例一直比较一般，主要是由于一些关键的技术点无法突破，技术开发组有更高的技术能力，是否能安排进行协助一些技术点的突破和预研。希望自动化组更多的能做一些技术指导和攻坚。
	我们后端智能服务器组业务重点是云中心和边缘域的服务器设备，业务重心的ISAPI协议测试可以跟涛涛那边配合搞起来，但是现在有一些设备的三个网口的断网，网络配置异常情况，设备断电，设备风扇异常情况，加密狗插拔，需要我们耗人时到楼上测试间处理，并且现在深思框架测试体量庞大，这样的非侧重点测试希望能够自动化掉。
	我们组的产品大部分和显示相关, 但是显示无法做到自动化, 识别不出图像异常(花屏\卡顿\闪屏\帧率不足等),有没有办法对输出图像进行识别处理(比如大数据识别图片特征,或者抓码流分析等)
	设备硬件环境比较复杂,自动化难度比较高,比如稳定的修改PC或者DP服务器的分辨率等
	按键工装上位机工具需求
培训学习方面的需求：
1、自动化/”雨燕”培训：  主要是研发的集成测试人员；
2、python深度学习：	  部门个别自动化开发人员
3、写自动化用例（HITA），发现对HITA的一些基本语法不熟悉的话，往往很难进行下去，主要是一些和设备功能不相关的关键字，比如如下这些：
        : FOR
Exit For Loop If
Run Keyword And Return Status
Run Keyword And Continue On Failure
         主要是不知道有这些关键字可以用
   
       所以希望在HITA编写用例的基础培训中，增加一些这方面的用法举例或形成文档可以参考

其他
缺陷数据预测。基于已有的缺陷、产品成熟度、人员等因素，推断预期缺陷情况
何丽昕：
1、 大家当前工作中遇到的最大痛点是什么？（包括但不限于自动化。跟工作效率提升
相关的痛点，都可以反馈。我们看看能否尽一点绵薄之力，助力大家实现2019年的目
标）//自动化工具与项目的结合是一个痛点，现在工具非常多，真的好用能用的不是很多，建议先梳理现有工具，冗余的去掉，精华的留下，然后再收集需求做新工具
 
2、 大家对自动化技术组提供的相关产品与服务，有什么意见和建议，可以畅所欲言
（比如：HITA框架、HITA官网、Tacos系统、自动化人才培养项目“雨燕”、以及跟产
品组合作的自动化改进项目等）//这些都挺好的，HITA在逐步的更加稳定，雨燕也培养了很多人出来，没啥建议了，挺好的

朱小龙7：
我这边根据自身工作情况希望2019年能够获得自动化技术组的帮助
1、	现象：基线产品不断有新的接口，而这些接口我们HITA中又没有，所以要依靠自动化组帮忙添加；
影响：这种情况下很费事，很影响工作效率；
建议：能否有相关编写HITA的培训或者开发帮助文档，对于一些HITA中未包含的简单接口可以自行解决，这样灵活性也会高些
2、‘雨燕计划’是只有自动化技术组才可以参加么？我们可不可以参加？或者我们可以获得相关培训文档么？

盛皓云
目前公司测试工作存在以下问题：
1、	项目周期短
2、	项目排期紧凑
3、	项目变更多
4、	产品线多，且差异性较大

像我们报警业务组还存在以下问题：
1、	设备外接的模块多，且通过无线或有线（导线）形式连接，不得不结合各种工装，如果采用自动化，在环境搭建和参数配置上耗费大量时间，且设备和模块的交互没法很好的结合自动化测试。
2、	可自动化率较低，维护又需要投入较大工作量，效率提升不明显，而且存在后续维护人员工作量投入的问题。目前组内已无人力能够投入自动化运维。
3、	写完的自动化用例基本只能在同个项目的维护项目上应用，产品线不同产品较多，差异大，可能写完以后就没用了或者需要进行较大调整，可重复利用率低。
以上导致自动化工作没法很好的展开。

是否考虑在测试部范围内，成立专门的自动化测试和维护小组，负责可自动化的用例测试，并负责自动化用例维护，接入不同小组的自动化测试，将手工测试与自动化测试分离开。



郑琪6

         门禁报警产品测试组目前遇到的问题反馈如下：
         1、2019年有个按键工装的上位机软件需要你们帮忙开发，需求相关可以联系胡铖涛工或者找我。
         2、2019年EHome协议测试较多，能否帮我们开发自动化测试，提高测试效率，类似x-men平台测试ISAPI协议那样（若ET组有规划，你们可忽略）。
         3、 Tacos系统，目前主要是用于组内工具发布，能否实现需求提交、缺陷提交、文档评审、缺陷统计等管理工具开发的各个生命周期。当前issue可以管理，但是无法统计数据，领导想看工作量，类似ET数据，Tacos能否实现？


谢天
数据资源平台部-质量保证部/性能与自动化测试组

最大的困惑是自动化测试和手动测试的分离。我一直觉得自动化测试只是一种测试技术，不应该作为一个独立的职位，问题是实际上自动化独立成组为其他项目组服务，
根据其他项目组的需求进行用例自动化和工具开发，职责不同导致关注点不一致，信息沟通和交流也会出现隔阂，产生误解。我的观点是手动测试和自动化测试成为一个组来对接项目，对项目质量负责，任何自动化技术最终也都是为了保证项目质量。
         
然后在技术方面，我想问下 在功能测试中除开围绕测试用例的用例自动化、用例管理系统、接口测试工具等，有没有一些前沿实用的测试自动化的思想和技术的分享和代码实践



陈欢
1.由于技术开发组更多是在平台角度考虑，所以很多时候没有考虑到产品使用的一些便捷性或者使用要求，一些平台搭建的时候，最好能预先收集下产品这边的需求，早期可能考虑不够，如T-LAB搭建了很久了，但使用率一直不是很高，近期考虑了产品组需求建设的X-man就能很快的利用起来，说明效果还是很明显的。
2.针对于我们对于各个平台的需求，以及这些需求的规划，好像没有成为一个闭环，我们不知道哪些需求是被接受了，会在哪个版本实现等等信息，你们发布的各个版本具体增删改了什么东西，我们也不是很清楚。
3.跟随产品组的自动化支撑感觉没有流程。产品组现在其实有很多自动化的需求，但不知道往谁那边提，怎么支持，感觉整体的接口不是很清晰，大部分就是靠口头去找，去沟通。
4.是否能协助产品组进行一些技术上的突破，产品组测试这边整体的自动化比例一直比较一般，主要是由于一些关键的技术点无法突破，技术开发组有更高的技术能力，是否能安排进行协助一些技术点的突破和预研。希望自动化组更多的能做一些技术指导和攻坚。
5.HITA本身的问题，HITA本身执行的时候发现由于脚本语言，运行速度较慢，此外有些脚本每次执行反馈的结果会不一样（比方有时候可以通过有时候无法通过），也没有发现问题，HITA整体并发处理能力较弱，长时间稳定性较弱，等等想看下是否有这些优化这些已有问题的计划。以上问题导致我们非用例的自动化已经慢慢转为用python了。
6.TACOS现在更多的只用在工具发布，建议除了工具发布也可以作为一个技术论坛使用，比方可以解决一些技术问题，技术分享等等，用于建立和培养部门内的技术氛围，感觉现在是缺少这样一个东西的。
7.更希望你们能提供的是一个可持续进行技术或自动化发展的框架，具体的实现则由各产品组本身实现。
8.可能和自动化组无关，也提一下，有时候技术开发组出发点好的，想兼容各个产品组，兼容各个功能，但有时候做的产品可能大而全，会导致上线会慢，为了兼容会导致各个产品组用起来都不那么顺畅。


俞俊6
1.	HITA安装路径改在C盘后，占用很多存储空间，目录下还有一些以日期结尾的备份hita, 工作机C盘本身只有100G会造成存储不足  （HITA的备份文件的大小为1个G，确实太大了）
2.	部分海外纯英文操作系统无法离线安装HITA包，需要手动解压，手动修改环境变量等。 （英文版没详细考虑，兼容性存在问题。后续专门出英文版本）

王梦茹
很荣幸可以参与到这次的讨论中，我们是报警产品业务部，集成测试小组成立不久，用到的公司产品主要是HITA，遇到的问题主要有以下几点：
1、	HITA的安装：一键安装HITA容易出现问题，且库文件导入后会出现无法识别关键字的情况，由于安装过程中体验感不好，大家都不想用；
建议：对于HITA安装过程中出现过的问题进行搜集，整理成Q&A，方便大家参考
2、	HITA调用接口的速度慢，同样一个接口，使用HITA要10s才会返回结果，使用Jmeter 1s内返回结果
建议：优化HITA处理接口的速度
3、	密码的加密算法，没有关键字可用（ISAPI协议中），我们自己用python2写了关键字但是用不了…
建议：针对公司内通用的的几种加密方式进行统计生成关键字，在使用HITA时，对于用户登录/新增时的密码加密问题，这也是我们目前使用工具进行接口测试时的一个难点
4、	SDK库的维护不及时，在使用过程中发现有些接口没有对应的关键字可调用
建议：希望可以加强对SDK的关键字的维护，及时进行增改
              
以上问题都比较具体，但是影响到我们使用的几个重要问题，我们目前的测试团队人员比较少，很多测试工作需要开发加入，所以测试工具安装方便、使用易上手对我们来说是比较重要的。

也希望针对公司的自动化工具，可以录制一些视频培训课程，让大家学习学习工具的使用。
       
              在使用HITA的过程中，感谢陈奕如陈工多次的耐心指导，很多问题都得到了解决。

陈咿米
测试改进设想文档

鲁之冬
我是传输显示—信息发布产品组的。
本组产品很多是安卓终端，目前技术开发组涉及安卓自动化的较少，之前参加雨燕培训也很少相关课程。
本组安卓自动化一直是摸索阶段，主要用的是MonkeyRunner和Appium，实现功能都是写的单独的py脚本, 也未进行汇总合并
不知是否有更好的进行更系统化的自动化的建议和思路？
盛叶
自动化开发的想法：
1.自动化开发的过程很繁琐：ET上根据用例生成脚本->线下编写脚本->本地调试->上传svn->配置测试环境->ET上自动执行
整个过程需要切换ET、本地、svn；用例变动整合脚本比较麻烦；脚本变动，et执行过程也复杂；
希望后续把整个过程移至线上，都能打通；
2.自动化测试环境
1）T-lab运行虚拟机环境不够稳定，之前用了几次，创建经常失败；其他操作用户体验也不是很好（操作异常无明确反馈）；
2）调试环境和实际执行环境不一致，导致的重复调试；
3．自动化测试框架
1）关键字执行效率低，尤其以cs，bs为重，cs关键字单执行耗时平均3s，bs关键字必须每次都要新打开浏览器，调试效率很低；
2）RIDE框架不适合编写代码逻辑稍微复杂的用例，效率比利用python直接开发更低；
4．自动化开发规范
经过开发实践，个人觉得 用例-步骤-操作-变量，可以简化为 用例-关键字-变量

工作角色上的想法：
根据以往的经历来看，技术开发组对于各个产品组的需求实现响应周期是比较长的；此外各组也成立了项目支持组开展相关的技术性工作；
个人认为技术开发组的角色：承担整个测试部的基础服务建设、统领技术点、实现不紧急且重要，实现周期允许较长的技术需求点；
各产品组项目支持组：承担产品组的基础服务建设、技术预研、推进技术应用；

其他可改进的想法：
1.	测试资源的利用，目前测试部的资源使用主要实现借用信息登记；而资源是否得到合理利用，各组资源分布情况，资源的线下借用流程简化，查询便捷等还有改进空间；（之前有同事反馈可以做一下库房管理）
2.	数据展示，可结合ET项目登记情况展示某时间段内的甘特图和任务量，便于各组评估分配人力投入；（同事反应研发也希望根据测试部的测试安排及时调整提测情况）
3.	任务管理，组内对任务这块的实现情况需求还比较大，了解到已有相应计划去优化任务管理功能；希望后续能积极应用起来；
4.	缺陷提交，可考虑提供给测试人员更便捷的服务，如问题截图，操作录制，缺陷信息获取，沟通记录保存等；

赵靖阳
我们后端智能服务器组业务重点是云中心和边缘域的服务器设备，业务重心的ISAPI协议测试可以跟涛涛那边配合搞起来，但是现在有一些设备的三个网口的断网，网络配置异常情况，设备断电，设备风扇异常情况，加密狗插拔，需要我们耗人时到楼上测试间处理，并且现在深思框架测试体量庞大，这样的非侧重点测试希望能够自动化掉。

组内之前没有涉及HITA自动化测试的业务，未来这部分是否能够通过HITA实现？鱼工如觉得技术上容易做，我们可以探讨一下方案。

wuhaitian 
1.	关于自动化的痛点，个人认为当前来看更多的还是在落地性以及使用效果上。自动化本身的目的是替代一部分的手工测试，进而提高我们的测试效率。 关于比较基础的自动化用例编写，在各位的推广下，在测试部掌握相关技术的人已经越来越多了。但如何选取可靠有效的自动化编写策略。编写完成后如何在项目中持续有效的应用。或者在一些特殊情况下（如业务接口变化比较频繁，导致维护成本比较高，但项目又需要自动化）的场景下如何应对。包括后续的如何评价自动化的质量等等，一些内容，个人感觉是需要深入项目中，结合项目实践才能整理出一些方案，因此如果能收集各个组的需求，对不同的问题，选取试点项目，参与进来后，做一些试点方案，如果有效，再推广到其他项目或形成方案。这样，是否会更贴近实际一些。
2.	印象中，技术开发组曾经推广过一个论坛性质的平台。现在的自动化编写中，据我了解还是存在由很多问题，但在自动化技术组或比较有经验的自动化编写同事来看，其实是比较简单的问题，能否真正有意义，有效果的把类似提问解答的平台或者说是部分氛围构建起来，让大家的问题能得到快速解答，是否也算是一种技术组对外提供的支持。当然，如果能建立起来，个人觉得不再只是一个自动化提问解答平台，相对来说问题的核心还是这种文化氛围构建。


fengxiaodan
1).我们组的产品大部分和显示相关, 但是显示无法做到自动化, 识别不出图像异常(花屏\卡顿\闪屏\帧率不足等),有没有办法对输出图像进行识别处理(比如大数据识别图片特征,或者抓码流分析等)
2).自动化用例不稳定或者用例执行环境复杂,质量不够好, 导致成员不爱用
3).设备硬件环境比较复杂,自动化难度比较高,比如稳定的修改PC或者DP服务器的分辨率等


邓名杰
之前的自动化主要围绕反复操作、压力及少部分的功能，2019也希望有所突破，也希望和技术开发组更好地对接解决业务组的问题。
2019年如何提效是传输测试这边的重中之重。

xuchaomin
是否可以开个python深度学习方面的课程？   （考虑一起加入学习，暂无计划，水平不够）

huangchanchan
1.	UI自动化维护成本高： 之前做的都是UI自动化，客户端界面变化了，之前写的脚本就要进行维护（成本较大），导致我们后面较少的使用自动化测试；是否有哪些改进方案能够优化
2.	脚本复用性不高，需要增加自动化使用率：目前实现的都是配置类的操作，但是配置类的操作只有大项目才会详细测试，维护版本都是确认性测试，脚本复用性不高；看看客户端其他模块是否也可以进行自动化测试
3.	提供人力支援底层库封装：目前项目业务较多，投入自动化改进较少；希望能够支援人力指导搭建客户端自动化底层框架
4.	接口测试：如何将项目业务和接口测试结合，目前只进行接口正向测试，测试意义不是太大；希望能够一起讨论一下，提供一下接口测试的方向

胡铖涛
按键工装上位机工具需求
闸机工装

吴萌
我在近3个月内，投入了一些时间写自动化用例（HITA），发现对HITA的一些基本语法不熟悉的话，往往很难进行下去，主要是一些和设备功能不相关的关键字，比如如下这些：
        : FOR
Exit For Loop If
Run Keyword And Return Status
Run Keyword And Continue On Failure
         主要是不知道有这些关键字可以用
   
       所以希望在HITA编写用例的基础培训中，增加一些这方面的用法举例或形成文档可以参考

马伯玲
反馈的主要是跟ET系统相关的问题，已经转发涛涛处理

刘跃
我们这边有没有缺陷数据预测这个东西呢？ 可以根据人员、产品新旧程度等推断出预期缺陷情况，应用机器学习的方法进行数据预测，不知道我们的数据够不够，哈哈。
         如果可以，请采纳哦，公司在朝着AI进行转型，哈哈，我们也可以提取一些场景进行转型

