import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    

    warped_image = np.array(image)
    warped_image[:,:]=[255,255,255]
    source_pts=source_pts[:,::-1]
    target_pts=target_pts[:,::-1]
    ### FILL: 基于MLS or RBF 实现 image warping
    #使用MLS
    for (y,x) in np.ndindex(warped_image.shape[:2]):
        print(y,x)


        if (y,x) in source_pts:
           for i in range(len(source_pts)):
               if np.array_equal(source_pts[i], np.array([y, x])):
                   warped_image[target_pts[i][0],target_pts[i][1]]=image[y,x]
                   break
        else:
            w=[1.0/((source_pts[i][0]-y)**2+(source_pts[i][1]-x)**2) for i in range(len(source_pts))]  
            
            p = np.sum([source_pts[i] * w[i] for i in range(len(source_pts))], axis=0) / np.sum(w)
            
            q = np.sum([target_pts[i] * w[i] for i in range(len(target_pts))], axis=0) / np.sum(w)
            
            M=np.linalg.inv(np.sum([w[i]*np.outer(source_pts[i]-p,source_pts[i]-p) for i in range(len(source_pts))],axis=0))@np.sum([w[i]*np.outer(source_pts[i]-p,target_pts[i]-q) for i in range(len(source_pts))],axis=0)
            (y1,x1)=np.dot(np.array([y,x])-p,M)+q
            if 0 <= int(y1) < warped_image.shape[0] and 0 <= int(x1) < warped_image.shape[1]:
                warped_image[int(y1),int(x1)]=image[y,x]
        
        #去掉白色细线
    newimage=warped_image.copy()
    for (y,x) in np.ndindex(newimage.shape[:2]):
        print(y,x)
        if np.array_equal(newimage[y,x], np.array([255,255,255])):
            if 2<=y<=warped_image.shape[0]-3 and 2<=x<=warped_image.shape[1]-3:
                if  not np.array_equal(warped_image[y,x-1], np.array([255,255,255])): 
                    newimage[y,x]=warped_image[y,x-1]
                if  not np.array_equal(warped_image[y+1,x], np.array([255,255,255])): 
                    newimage[y,x]=warped_image[y+1,x]
                elif not np.array_equal(warped_image[y-1,x], np.array([255,255,255])): 
                    newimage[y,x]=warped_image[y-1,x]
                elif not np.array_equal(warped_image[y,x+1], np.array([255,255,255])): 
                    newimage[y,x]=warped_image[y,x+1]
                elif  not np.array_equal(warped_image[y,x-1], np.array([255,255,255])): 
                    newimage[y,x]=warped_image[y,x-1]
                elif  not np.array_equal(warped_image[y,x-2], np.array([255,255,255])): 
                    newimage[y,x]=warped_image[y,x-2]
                elif  not np.array_equal(warped_image[y,x+2], np.array([255,255,255])): 
                    newimage[y,x]=warped_image[y,x+2]
                elif  not np.array_equal(warped_image[y-2,x], np.array([255,255,255])): 
                    newimage[y,x]=warped_image[y-2,x]
                elif  not np.array_equal(warped_image[y+2,x], np.array([255,255,255])): 
                    newimage[y,x]=warped_image[y+2,x]
                    
                

            

    return newimage

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()