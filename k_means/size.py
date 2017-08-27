import csv 

print("hello")
input_file = open("train.csv","r")
reader_file = csv.reader(input_file)
value = len(list(reader_file))
print("no of rows in file: ",value)