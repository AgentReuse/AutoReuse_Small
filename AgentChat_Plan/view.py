from diskcache import Index

cache = Index('./semantic_cache')  # 路径与项目中的保持一致

print("当前缓存内容如下：\n")
for key in cache:
    print(f" Key: {key}")
    print(f" Value: {cache[key]}")
#cache.clear()