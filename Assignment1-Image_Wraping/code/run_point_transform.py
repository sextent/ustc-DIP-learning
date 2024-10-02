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
    print(x,y)
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
def point_guided_deformation(image, source_pts, target_pts, alpha=0.8, eps=1e-8):
    print(source_pts)
    # 获取图像尺寸
    h, wide = image.shape[:2]
    print(h,wide)
    # 创建空白的变形图像,进行复制
    warped_image = np.zeros_like(image)
    # 生成网格坐标
    grid_x, grid_y = np.meshgrid(np.arange(wide), np.arange(h))
    grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    # 初始化新位置数组
    new_positions = np.zeros_like(grid_points, dtype=int)
    print(new_positions.shape)
    print(new_positions[1])
    #print(new_positions[1, 0])
    #print(new_positions[1:10, 0])
    # 计算权重函数
    def compute_weights(v, p, alpha):
        dist2 = np.sum((v - p)**2, axis=1) + eps
        w = dist2 ** (-alpha)
        return w
    
    # 计算每个像素点的变形
    for idx, v in enumerate(grid_points):
        # 跳过与 source_pts 重合的点
        if np.any(np.all(np.isclose(v, source_pts), axis=1)):
            continue  # 如果 v 和 source_pts 中的任意点相等，跳过该点
        # 计算权重
        w = compute_weights(v, source_pts, alpha)
        # 计算加权质心
        w_sum = np.sum(w)
        if w_sum>1:
            print('beside point',v,w_sum)
        p_centroid = np.sum(w[:, np.newaxis] * source_pts, axis=0) / w_sum
        q_centroid = np.sum(w[:, np.newaxis] * target_pts, axis=0) / w_sum
         # 计算去中心化的坐标
        p_hat = source_pts - p_centroid
        q_hat = target_pts - q_centroid
        #if idx<10:
            #print(p_centroid,q_centroid)
            #print(p_hat,q_hat)
        # 构建矩阵
        p_hat_T = p_hat.T
        w_diag = np.diag(w)
        M = p_hat_T @ w_diag @ p_hat
        escape_point=0
        #if np.linalg.det(M)>0.01:
            #print(np.linalg.det(M))
         # 如果矩阵 M 不可逆，跳过该点
        if np.linalg.det(M) < eps:
            new_positions[idx] = v
            escape_point=escape_point+1
            # print('escape')
            continue
        #print('escape_point is:',escape_point)
        # 计算仿射矩阵 A,np.linalg.inv为求逆操作
        M1 = np.linalg.inv(M) @ (p_hat.T @ w_diag @ q_hat) 
        A = M1
        # 计算新位置
        v_hat = v - p_centroid
        new_v = v_hat @ A + q_centroid
        new_positions[idx] = new_v
        # 将新位置映射回图像坐标系
        # 对新坐标取整数
        new_positions[idx] = np.round(new_v).astype(int)  # 先取整，再转换为整数
        # 边界检查，确保在合法范围内
        # 原始位置v
        if 0 <= new_positions[idx][0] < wide and 0 <= new_positions[idx][1] < h:
            # 逐个像素替换
            warped_image[new_positions[idx][1], new_positions[idx][0]] = image[v[1], v[0]]
    return warped_image

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