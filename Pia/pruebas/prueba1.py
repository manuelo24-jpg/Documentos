a, b = 0,1
while a < 10:
    print(a)
    a, b = b, a+b
print("El primer elemento mayor que 10 es ", a)