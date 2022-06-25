from pathlib import Path
import re
import json

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

    try:
        q = i.split("}")[1].split("\\")[0].strip()
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
    dest_csv_name = curr_tex.split("/")[-1].replace(".tex", ".json")
    #curr_res_csv = open(dest_csv_path+dest_csv_name, 'w+')
    #curr_res_csv.write("sourceTask,destinationTask,value\n")

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

    # {i: 5 for i in listOfStr}
    json_res = {"Full": {"Frozen": {i:{"T2":"", "T4":"", "T6":""} for i in src_dict},
                         "Unfrozen": {i:{"T2":"", "T4":"", "T6":""} for i in src_dict}},
                "Limited": {"Unfrozen": {i:{"T2":"", "T4":"", "T6":""} for i in src_dict}}}

    curr_res = []
    for i, src in enumerate(src_dict):
        for k, dest in enumerate(dest_tasks):
            dest = re.findall('[A-Z][^A-Z]*', dest)
            json_res[dest[0]][dest[1]][src][dest[2]] = src_dict[src][k]


    json_res = json.dumps(json_res)
    json_file = open(dest_csv_path+dest_csv_name, 'w+')
    json_file.write(json_res)



#print()








