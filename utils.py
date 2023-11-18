import re

def split_string(input_string):#分隔函数内的参数，以逗号划分，不能分隔(),[],{},""这四种括号内的逗号
    pattern = r',(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\]|[^\(]*\)|[^\{]*\})'
    result = re.split(pattern, input_string)
    return result

def check(code_str,function_lst):#检查该字符串是否存在函数调用（匹配开头）
    for func in function_lst:
        funcpattern=re.compile(func+"\((.*)\)",re.DOTALL)
        result=re.match(funcpattern,code_str)
        if result:
            lst=split_string(re.findall(funcpattern,result.group(0))[0])
            lst=[q.strip() for q in lst]
            return [func,lst]
    return False


def make_lst(code_str,function_lst,all_lst,record,recorddic):#将原始函数调用链转为列表格式
    #all_lst=[["Oregon","2023-10-15"],[["Oregon","2023-10-15"],[12,15]],13]生成这种类型的数据

    result = check(code_str,function_lst)

    if result:
        #最外层括号的下一层参数
        recorddic[record]=result[0]
        result=result[1]
        for i in range(len(result)):
            if not check(result[i],function_lst):
                all_lst.append(result[i])
            else:
                all_lst.append([])
                if record=="":
                    make_lst(result[i],function_lst,all_lst[i],record+str(i),recorddic)
                else:
                    make_lst(result[i],function_lst,all_lst[i],record+","+str(i),recorddic)

    return all_lst,recorddic

def modify_nested_list(nested_list, indices, new_value):#step-2:从嵌套最底层的列表向上递归地修改
    if len(indices) == 0:
        pass
    if len(indices) == 1:
        nested_list[indices[0]] = new_value
    else:
        modify_nested_list(nested_list[indices[0]], indices[1:], new_value)

