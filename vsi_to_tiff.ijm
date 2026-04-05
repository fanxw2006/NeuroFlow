// --- vsi_to_tiff_converter.ijm ---
// 功能：批量将 Olympus .vsi 多�?�道图像转换为单通道�? TIFF 文件�?
// 流程：�?�择输入/输出文件�? -> 循环处理 .vsi 文件 -> 打开指定层级 -> 分�?�道 -> �?8bit -> 自动增强对比�? -> 按�?�道命名并保存�??

// 1. 用户交互：�?�择输入和输出文件夹
dir = getDirectory("Choose a Directory of input");


outputdir = getDirectory("Choose a Directory of output");


// 2. 用户交互：获取输出文件名前缀
name = getString("Type in output file name prefix", "PdynVMH_RV_slide1");

// 3. 获取文件列表并设置批处理模式
list = getFileList(dir);
setBatchMode(true); // 运行速度更快，不显示图像窗口
setOption("BlackBackground", true); // 确保新生成的图像背景为黑�?

// 4. 主循环：遍历文件夹中的所有文�?
for (f = 0; f < list.length; f++) {
    path = dir + list[f];
    showProgress(f, list.length); // 显示进度�?

    // 筛�?�：只处�? .vsi 文件，并且排除名�? "Image_01_Overview.vsi" 的概览图
    if ((endsWith(list[f], ".vsi")) && (list[f] != "Image_01_Overview.vsi")) {
        
        // 从文件名中提�? section 编号 (例如�? "Image_02_Section01.vsi" 中提�? "01")
        // 注意：这依赖于固定的文件名格�? "Image_XX_SectionYY.vsi"
        section = substring(list[f], indexOf(list[f], "Section") + 7, indexOf(list[f], ".vsi"));
        print("正在处理: " + list[f] + " (Section: " + section + ")");

        // 5. 核心处理步骤

        // a. 打开 .vsi 文件的特定层�? (Group1, Level1)
        // 这依赖于 Olympus �? OVMacro 插件
        run("Bio-Formats Importer", "open=[" + path + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT");

        // b. 分割通道
        run("Split Channels");
        
        // c. 循环处理每个分割出的通道
        for (c = 1; c <= nImages; c++) {
            selectImage(c);
            
            // d. 将�?�道转为 8-bit
            run("8-bit");
            
            // e. 自动增强对比�? (Auto Scaling)
            run("Enhance Contrast...", "saturated=0 normalize");
            
            // f. 构建保存路径和文件名
            // 格式：[输出前缀]_[Section编号]_C[通道号].tif
            // 例如：PdynVMH_RV_slide1_01_C1.tif
            save_path = outputdir + name + "_" + section + "_C" + c + ".tif";
            
            // g. 保存�? TIFF
            saveAs("Tiff", save_path);
            
            // h. 关闭当前通道图片
            close();
        }
    }
}

// 6. 处理结束
setBatchMode(false);
print("批处理完成！�?有文件已保存�?: " + outputdir);
beep; // 发出提示�?
