from pathlib import Path


def extract_num(i):
    try:
        q = i.split("}")[1].split("$")[0].split("{")[1].strip()
        try:
            qq = float(q)
            return q
        except:
            pass
    except:
        pass

    try:
        q = i.split("$")[0].strip()
        try:
            qq = float(q)
            return q
        except:
            pass
    except:
        pass

    try:
        q = i.split("{")[1].split("}")[0].strip()
        try:
            qq = float(q)
            return q
        except:
            pass
    except:
        pass

    try:
        q = i.strip()
        try:
            qq = float(q)
            return q
        except:
            pass
    except:
        pass

    try:
        q = i.split("\\")[0].strip()
        try:
            qq = float(q)
            return q
        except:
            pass
    except:
        pass

    raise Exception



# iterate tex files
for curr_tex in list(map(lambda x: "./"+str(x), list(Path(".").rglob("*.[tT][eE][xX]")))):
    print(curr_tex)
    dest_csv_path = "/".join(curr_tex.split("/")[:-1])+"/"
    dest_csv_name = curr_tex.split("/")[-1].replace(".tex", ".csv")
    curr_res_csv = open(dest_csv_path+dest_csv_name, 'w+')
    curr_res_csv.write("sourceTask,destinationTask,value\n")
    curr_tex = open(curr_tex).read().split("toprule\n")[1].split("\n\\bottomrule")[0].split("\n\midrule")
    dest_tasks = list(map(lambda x: x.replace("\\\\", '').strip(), curr_tex[0].split("&")[1:]))
    src_tasks = list(map(lambda x: x.split("&"), curr_tex[1].split("\n")[1:]))
    src_dict = {}

    for j, src in enumerate(src_tasks):
        try:
            task_name = src[0].split("{")[1].split("}")[0].strip()
        except:
            task_name = src[0].strip()

        src_dict[task_name] = []

        for i in src[1:]:
            num = extract_num(i)
            src_dict[task_name].append(num)

    curr_res = []
    for i, src in enumerate(src_dict):
        for k, dest in enumerate(dest_tasks):
            for j, res in enumerate(src_dict[src]):
                print(k, i, j, dest, src, res)
                curr_res.append([k, j, dest, src, src_dict[src][k]])
                break

    curr_res = "\n".join(list(map(lambda x: ",".join(x[2:]), sorted(curr_res, key=lambda x: (x[0], x[1], x[2])))))
    curr_res_csv.write(curr_res)
    #print()



#print()








