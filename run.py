# coding=utf-8
import os
import json
import re
from tqdm import tqdm
import openai
from utils import split_string,check,make_lst,modify_nested_list

openai.api_key = "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"

field_path = "./api_set.jsonl"

save_raw_data_path="./rawdata_pot.jsonl"
api_path = "./api_pot.jsonl"
function_call_path="./function_call_pot.jsonl"

two_step_raw_data_path="./two_step_rawdata_pot.jsonl"
two_step_function_call_path="./two_step_function_call_pot.jsonl"

field_file=open(field_path,"r",encoding="utf8")
save_raw_data_file=open(save_raw_data_path,"w",encoding="utf8")
api_file=open(api_path,"w",encoding="utf8")
function_call_file=open(function_call_path,"w",encoding="utf8")

two_step_raw_data_file=open(two_step_raw_data_path,"w",encoding="utf8")
two_step_function_call_file=open(two_step_function_call_path,"w",encoding="utf8")

prompt="Please generate a task in the given specific domain, and accomplish the task by generating some python functions with definitions and calling statements.\nPlease do not add additional assignment statements in `main()`, just generate a one-line function calling with arguments from the Task. The former function's results should be latter ones' input arguments (e.g. func1(func2(*args, **kwargs), func3(*args, **kwargs))).\nYou need to check if the function calling use values that are not present in the task, incorporate those values into the original task, rewrite the original task to ensure that it includes all the function calling values.The rewritten task should be written as a comment inside the main function.\nThe code cannot reference any existing python library.\nDon't use python's intrinsic functions in your code, such as lists and dictionaries (index access, slicing, adding, deleting, etc.). \nDon't invovle \"for\" or \"while\" loops in your code.\nThe parameter description only needs to be a sentence and does not require additional structured explanations.\n\nHere are some examples.\n\nDomain: Air Pollution\n\nAnswer:\nTask: Please help me analyze the PM2.5 level in Oregon on October 15, 2023, and send the result to the employees in Oregon.\n```python\ndef analyze_PM25(area: str, date: str) -> float:\n    \"\"\"Analyze PM2.5 status in the given area.\n\n    Args:\n        area (str): the name of the area\n        date (str): the date of the analyzation(e.g., yyyy-mm-dd)\n    \n    Returns:\n        pm25_result (float): the PM2.5 result of the area\n    \"\"\"\n    ...\n\n\ndef get_employee_list(area_name:str) -> list[int]:\n    \"\"\"Get the employee IDs for the area.\n    \n    Args:\n        area_name (str): the name of the area\n    \n    Returns:\n        employee_ids (list[int]): a list of employee IDs for the area\n    \"\"\"\n    ...\n\n\ndef send_result(result: float, receiver_ids: list[int]) -> bool:\n    \"\"\"Send the result to the receivers.\n\n    Args:\n        result (float): the result to be sent\n        receiver_ids (list[int]): a list of IDs to send the result to\n    \n    Returns:\n        status (bool):the status of the send, True if sent successfully, otherwise False\n    \"\"\"\n    ...\n\n\ndef main():\n    send_result(\n        analyze_PM25(\"Oregon\", \"2023-10-15\"),\n        get_employee_list(\"Oregon\")\n    )\n    #Please help me analyze the PM2.5 level in Oregon on 2023-10-15, and send the result to the employees in Oregon.\n\n\nif __name__ == \"__main__\":\n    main()\n```\n\nDomain: Weather Forecast\n\nAnswer:\nTask: Please help me retrieve the weather forecast for New York City on November 5, 2023, and notify the residents.\n```python\ndef get_weather_forecast(city: str, date: str) -> str:\n    \"\"\"Retrieve the weather forecast for a specific city and date.\n\n    Args:\n        city (str): the name of the city\n        date (str): the date for the weather forecast(e.g., yyyy-mm-dd)\n\n    Returns:\n        forecast (str): the weather forecast for the specified city and date\n    \"\"\"\n    ...\n\n\ndef get_resident_emails(city: str) -> list[str]:\n    \"\"\"Get the email addresses of residents in the city.\n\n    Args:\n        city (str): the name of the city\n\n    Returns:\n        email_list (list[str]): a list of email addresses of residents\n    \"\"\"\n    ...\n\n\ndef notify_residents(email_list: list[str], forecast: str) -> bool:\n    \"\"\"Notify the residents with the weather forecast.\n\n    Args:\n        email_list (list[str]): a list of email addresses of residents\n        forecast (str): the weather forecast to be sent\n\n    Returns:\n        status (bool): the status of the notification, True if successfully notified, otherwise False\n    \"\"\"\n    ...\n\n\ndef main():\n    notify_residents(\n        get_resident_emails(\"New York City\"),\n        get_weather_forecast(\"New York City\", \"2023-11-05\")\n    )\n    #Please help me retrieve the weather forecast for New York City on 2023-11-05 and notify the residents.\n\n\nif __name__ == \"__main__\":\n    main()\n```\n\n\nDomain: Robot Planning\n\nAnswer:\nTask: Please help me plan the route for a robot to go from location A to location B and then perform a task at location C.\n```python\ndef plan_route(start_location: str, end_location: str, task_location: str) -> list[str]:\n    \"\"\"Plan the route for a robot to go from start_location to end_location and then perform a task at task_location.\n\n    Args:\n        start_location (str): the starting location for the robot\n        end_location (str): the ending location for the robot\n        task_location (str): the location where the robot needs to perform a task\n\n    Returns:\n        route (list[str]): a list of locations in the planned route\n    \"\"\"\n    ...\n\n\ndef execute_task(route: list[str]) -> bool:\n    \"\"\"Execute the planned route and perform the task.\n\n    Args:\n        route (list[str]): a list of locations in the planned route\n\n    Returns:\n        status (bool): the status of the task execution, True if successfully executed, otherwise False\n    \"\"\"\n    ...\n\n\ndef main():\n    execute_task(plan_route(\"Location A\", \"Location B\", \"Location C\"))\n    #Please help me plan the route for a robot to go from location A to location B and then perform a task at location C.\n\n\nif __name__ == \"__main__\":\n    main()\n```\n\nDomain: Data Processing\n\nAnswer:\nTask: Please assist in calculating the average temperature from a set of temperature readings, and provide the results in a report.\n```python\ndef calculate_average_temperature(temperature_readings: list[float]) -> float:\n    \"\"\"Calculate the average temperature from a list of temperature readings.\n\n    Args:\n        temperature_readings (list[float]): a list of temperature readings\n\n    Returns:\n        average_temperature (float): the calculated average temperature\n    \"\"\"\n    ...\n\n\ndef generate_report(average_temperature: float) -> str:\n    \"\"\"Generate a report with the calculated average temperature.\n\n    Args:\n        average_temperature (float): the calculated average temperature\n\n    Returns:\n        report (str): a report containing the average temperature\n    \"\"\"\n    ...\n\n\ndef main():\n    report = generate_report(calculate_average_temperature([72.5, 68.3, 74.2, 70.8, 69.5]))\n    # Please assist in calculating the average temperature from a set of temperature readings, in sequence, 72.5, 68.3, 74.2, 70.8, 69.5, and provide the results in a report.\n\n\nif __name__ == \"__main__\":\n    main()\n```\n\n\nIt's your turn to generate. \n\nDomain: {}\n\nAnswer: __\n"

two_step_prompt="Below are some APIs, a task, and parameter-value pairs for this task. You need to check if any values of the pairs that are not present in the task, incorporate those values into the original task without mentioning the specific APIs, rewrite the original task to ensure that it includes all the values,which is very important.If the value is in the format of list, you need to incorporate each item from the value into the task in natural language, rather than directly adding the value in the format of list to the task.\nYou only need to generate the rewritten task and nothing else.\n\n"

data_id=0

sub_field_lst=[]
for field_dic in field_file:
    field_dic=json.loads(field_dic)
    raw_field=list(field_dic.keys())[0]
    subfield=re.findall("/(.*)",raw_field)[0]#去掉前缀后的子领域
    sub_field_lst.append(subfield)

for subfield in tqdm(sub_field_lst):
    data_id+=1
    try:
        new_prompt=prompt.format(subfield)
        messages=[]
        messages.append({'role': 'user', 'content': new_prompt})
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages
        )
        text = completion.choices[0].message['content']

        d=dict()
        d["id"]=data_id
        d["prompt"]=new_prompt
        d["response"]=completion
        save_raw_data_file.write(f"{json.dumps(d,ensure_ascii=False)}\n")
        save_raw_data_file.flush()

        pattern1=re.compile("```python(.*?)def main",re.DOTALL)#识别生成的所有函数
        function_code=re.findall(pattern1,text)[0].strip()

        pattern2=re.compile("main\(\):(.*?)#",re.DOTALL)#识别main函数里的调用链
        function_call_code=re.findall(pattern2,text)[0].strip()

        pattern3=re.compile("#(.*?)if __name__",re.DOTALL)#识别main函数里的task
        task=re.findall(pattern3,text)[0].strip()


        #识别生成的函数
        function_lst=re.split("\.\.\.",function_code)

        function_all=[]
        
        for t in function_lst:
            t=t.strip()
            function=dict()
            try:
                api_name=re.findall("def(.*?)\(",t)[0].strip()
                function["api_name"]=api_name
                function["api_description"]=re.findall(re.compile("\"\"\"(.*?)\n",re.DOTALL),t)[0].strip()
                function["parameters"]=dict()
                function["responses"]=dict()

                args_returns=re.findall(re.compile(re.escape(function["api_description"])+"(.*?)\"\"\"",re.DOTALL),t)[0].strip()

                if re.findall("Returns:",args_returns):
                    responses_raw=re.findall(re.compile("Returns:(.*)",re.DOTALL),args_returns)[0].strip().split("\n")
                    if re.findall("Args:",args_returns):
                        parameters_raw=re.findall(re.compile("Args:(.*?)Returns",re.DOTALL),args_returns)[0].strip().split("\n")      
                    else:
                        parameters_raw=[]
                else:
                    responses_raw=[]
                    if re.findall("Args:",args_returns):
                        parameters_raw=re.findall(re.compile("Args:(.*)",re.DOTALL),args_returns)[0].strip().split("\n")
                    else:
                        parameters_raw=[]


                if  parameters_raw!=[]:
                    for i in parameters_raw:
                        i=i.strip()
                        para_tp = re.findall("\(.*?\)(?=:)", i)[0]
                        tp_raw=re.findall("\(.*?\)",para_tp)[-1]
                        tp=tp_raw[1:-1]
                        index=re.search(re.escape(tp_raw),i).span()[0]
                        parameter_name=i[:index].strip()

                        des=re.findall(":(.*)",i)[0].strip()
                        function["parameters"][parameter_name]={"type":tp,"description":des}
                
                if  responses_raw!=[]:
                    for i in responses_raw:
                        i=i.strip()
                        para_tp = re.findall("\(.*?\)(?=:)", i)[0]
                        tp_raw=re.findall("\(.*?\)",para_tp)[-1]
                        tp=tp_raw[1:-1]
                        index=re.search(re.escape(tp_raw),i).span()[0]
                        response_name=i[:index].strip()
                
                        des=re.findall(":(.*)",i)[0].strip()
                        function["responses"][response_name]={"type":tp,"description":des}

                function_all.append(function)
            except:
                continue

        
        #将API写入文件
        api_dic=dict()
        api_dic["id"]=data_id
        api_dic[subfield]=function_all
        api_file.write(f"{json.dumps(api_dic,ensure_ascii=False)}\n")
        api_file.flush()


        function_order=[function_all[k]["api_name"] for k in range(len(function_all))]#记录上面生成的function的顺序

        function_parameter_order={i["api_name"]:list(i["parameters"].keys()) for i in function_all}#记录每个函数的参数顺序

        function_response_order={i["api_name"]:i["responses"] for i in function_all}#记录每个函数的回复顺序，后来没用到

        #识别函数调用链

        front_span=100
        first_func_name=""
        for func in function_order:
            funcpattern=re.compile(func+"\((.*)\)",re.DOTALL)
            a=re.search(funcpattern,function_call_code)
            if a:
                if a.span()[0]<front_span:
                    front_span=a.span()[0]
                    first_func_name=func

        function_call_code=re.findall(re.compile(first_func_name+".*",re.DOTALL),function_call_code)[0]

        lst,func_find_dic=make_lst(function_call_code,function_order,[],"",{})#得到目标的列表和对应的字典
        # example:
        # lst=[["Oregon","2023-10-15"],[["Oregon","2023-10-15"],[12,15]],13]
        # func_find_dic={"0":"analyze_PM25","1":"get_employee_list","1,0":"analyze_PM25","":"send_result"}

        sorted_dic = dict(sorted(func_find_dic.items(), key=lambda x: len(x[0]), reverse=True))
        compare_lst=[]#方便对比的中间结果
        put_response_lst=[]#放了response的最终结果
        response_index=0

        while True:
            key = list(sorted_dic.keys())[0]
            p=lst
            if key!="":
                for key_index in [int(q) for q in key.split(",")]:
                    p=p[key_index]

            new_api_call={}
            api_name=sorted_dic[key]
            new_api_call["api_name"]=api_name
            new_api_call["parameters"]={}
            para_lst= function_parameter_order[api_name]
            for i in range(len(para_lst)):
                try:
                    new_api_call["parameters"][para_lst[i]]=eval(p[i])
                except:#防止值为API_call_1这样的格式
                    new_api_call["parameters"][para_lst[i]]=p[i]

            if new_api_call not in [p[0] for p in compare_lst]:
                res="API_call_"+str(response_index)
                compare_lst.append([new_api_call.copy(),res])
                
                new_api_call["responses"]=[res]
                response_index+=1
                
                put_response_lst.append(new_api_call)

                del sorted_dic[key]
                if len(sorted_dic)==0:
                    break
                modify_nested_list(lst,[int(q) for q in key.split(",")],res)
                
            else:
                for s in compare_lst:
                    if s[0]==new_api_call:
                        res=s[1]
                        break
                del sorted_dic[key]
                if len(sorted_dic)==0:
                    break
                modify_nested_list(lst,[int(q) for q in key.split(",")],res)

        #将调用链写入文件
        d=dict()
        d["id"]=data_id
        d["task"]=task
        d[subfield]=put_response_lst
        function_call_file.write(f"{json.dumps(d,ensure_ascii=False)}\n")
        function_call_file.flush() 


        #到第二阶段修改Task的过程
        two_step_final_prompt=two_step_prompt+"APIs:\n"

        para_value_lst=[]
        for call in d[subfield]:
            for key,value in call["parameters"].items():
                if isinstance(value,list):
                    tt=True
                    for i in value:
                        if re.search(re.escape(str(i)),task):
                            continue
                        else:
                            tt=False
                            break
                    if not tt:
                        para_value_lst.append(str(key)+"="+str(value))
                else:
                    if not re.search("API_call_",re.escape(str(value))):
                        if not re.search(re.escape(str(value)),task):
                            para_value_lst.append(str(key)+"="+str(value))

        if para_value_lst!=[]:

            para_value=",".join(para_value_lst)

            for funcc in function_all:
                two_step_final_prompt+=str(funcc)+"\n"
            
            two_step_final_prompt=two_step_final_prompt+"\nTask:"+task+"\n\nparameter-value pairs:\n"+para_value+"\n\n"+"rewritten task:__"

            messages=[]
            messages.append({'role': 'user', 'content': two_step_final_prompt})
            twostep_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )
            text = twostep_completion.choices[0].message['content']

            #二阶段原始生成
            d=dict()
            d["id"]=data_id
            d["prompt"]=two_step_final_prompt
            d["response"]=twostep_completion
            two_step_raw_data_file.write(f"{json.dumps(d,ensure_ascii=False)}\n")
            two_step_raw_data_file.flush()

            #二阶段最后结果
            d=dict()
            d["id"]=data_id
            d["task"]=text
            d[subfield]=put_response_lst
            two_step_function_call_file.write(f"{json.dumps(d,ensure_ascii=False)}\n")
            two_step_function_call_file.flush()
        
        else:
            d=dict()
            d["id"]=data_id
            d["task"]=task
            d[subfield]=put_response_lst
            two_step_function_call_file.write(f"{json.dumps(d,ensure_ascii=False)}\n")
            two_step_function_call_file.flush()
    
    except:
        pass


        