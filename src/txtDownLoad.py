import urllib.request

# 设置代理（示例：HTTP 代理）
proxy_handler = urllib.request.ProxyHandler({
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
})
opener = urllib.request.build_opener(proxy_handler)
urllib.request.install_opener(opener)

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path = "../data/the-verdict.txt"
urllib.request.urlretrieve(url, file_path)