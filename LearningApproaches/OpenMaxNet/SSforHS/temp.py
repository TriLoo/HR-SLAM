def func():
    a = (x for x in range(10))
    return a


for i in range(3):
    b = func()
    print('b = ', b)
    for i in b:
        print(i)
