### 使用方式

1. **编译项目**  
   进入示例目录并执行编译脚本：
   ```bash
   cd examples/alltoallv_gmm_v2
   bash scripts/build.sh
   ```

2. **运行AllToAllV-GMM示例程序**  
   进入示例目录并执行运行脚本：
   ```bash
   cd examples/alltoallv_gmm_v2
   bash scripts/run.sh $device_list $expert_num
   ```

   - **参数说明**：
     - `device_list`: 指定用于运行的设备（NPU）编号列表，以逗号分隔。

     - `expert_num`: 指定专家数，未显示指定时默认值为4。

        示例：使用第0和第1个NPU设备运行2卡AllToAllV-GMM, 专家数为4（专家并行数为2）：
       ```bash
       # 修改运行参数配置时可能需要同步shape文件
       bash scripts/run.sh 0,1 4
       ```

   - **配置计算规模**：  
     矩阵形状参数（M、K、N）可在配置文件 `scripts/test_shapes.csv` 中进行设置。  
     修改该文件以定义测试用例的输入维度。

   - **约束限制**： 
     - 该算子需保证专家数为参与运算的NPU设备数的整数倍。
     - 共享内存空间需大于所有A矩阵所占空间之和。
     - AIV核数需大于或等于参与运算的NPU设备数。

