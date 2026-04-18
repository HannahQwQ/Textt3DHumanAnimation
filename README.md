一、项目简介（Overview）

本项目旨在构建一个从自然语言文本到三维人体动作的生成系统，实现语义描述与三维动态表现之间的映射。
系统以文本作为输入，通过多模块协同处理，最终生成具有语义一致性与时序连续性的三维人体动作，并支持可视化与结果导出。

关键词：文本驱动（Text-to-Motion）、动作生成（Motion Generation）、三维重建（3D Human Reconstruction）


二、系统架构（Pipeline）

<img width="1360" height="630" alt="pipeline" src="https://github.com/user-attachments/assets/ad228fb3-9cf8-42d3-9763-3725efbb5ffe" />

Text Input → Text Encoder → Motion Generator → 3D Human Reconstruction → Rendering


系统首先接收用户输入的文本描述，并对文本中的语义信息进行解析，其中包括人物外观描述与动作语义信息。随后，系统通过文本驱动动作生成模块，根据文本语义生成对应的人体动作序列，并以 SMPL 人体模型的姿态参数形式进行表示。与此同时，三维人体建模与表示模块负责构建可驱动的三维人体模型，并确定人体的形状参数以及对应的三维表示结构。在获得人体结构信息与动作序列之后，系统进一步通过动作驱动与视频渲染模块，将生成的动作序列应用到三维人体模型上，从而得到随时间变化的人体姿态，并通过渲染过程生成最终的三维数字人动作视频。


三、生成效果

<img width="1388" height="392" alt="image" src="https://github.com/user-attachments/assets/0a56f0d3-2658-4e1e-b466-070b227874e9" />

<img width="1373" height="397" alt="image" src="https://github.com/user-attachments/assets/36799bb3-49f1-459d-9d5c-912d9d1701c5" />


四、动作模型对比实验

<img width="1258" height="370" alt="image" src="https://github.com/user-attachments/assets/cd3de602-e49d-4cb0-bcde-f1e9bf410d72" />



