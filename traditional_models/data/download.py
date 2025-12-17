import os  # 导入操作系统模块
import zipfile  # 导入压缩文件模块
from torchvision.datasets.utils import download_url  # 导入下载URL工具
from topmost.utils.logger import Logger  # 导入日志记录器


logger = Logger("WARNING")  # 创建警告级别的日志记录器


def download_dataset(dataset_name, cache_path="~/.topmost"):  # 定义下载数据集函数
    cache_path = os.path.expanduser(cache_path)  # 展开用户路径
    raw_filename = f'{dataset_name}.zip'  # 构建压缩文件名

    if dataset_name in ['Wikitext-103']:  # 如果是Wikitext-103数据集
        # download from Git LFS.  # 从Git LFS下载
        zipped_dataset_url = f"https://media.githubusercontent.com/media/BobXWu/TopMost/main/data/{raw_filename}"  # 构建Git LFS下载URL
    else:  # 否则
        zipped_dataset_url = f"https://raw.githubusercontent.com/BobXWu/TopMost/master/data/{raw_filename}"  # 构建GitHub原始文件下载URL

    logger.info(zipped_dataset_url)  # 记录下载URL信息

    download_url(zipped_dataset_url, root=cache_path, filename=raw_filename, md5=None)  # 下载数据集文件

    path = f'{cache_path}/{raw_filename}'  # 构建文件路径
    with zipfile.ZipFile(path, 'r') as zip_ref:  # 打开压缩文件
        zip_ref.extractall(cache_path)  # 解压到缓存目录

    os.remove(path)  # 删除压缩文件


if __name__ == '__main__':  # 主程序入口
    download_dataset('20NG')  # 下载20NG数据集
