import pandas as pd
import core_utils



def generate_taint_spec(input,output):

    # EXPECTED INPUT FORMAT
    # library,class,method_name,args,return,classification
    df = pd.read_csv(input)

    # auxiliar files 
    gps_class_df = core_utils.get_gps_map()
    aosp_class_map = core_utils.get_aosp_map()
    
    stmts = []
    for _,row in df.iterrows():
        args = eval(row['args'])
        if args[0] == '':
            args_fill = ''
        else:
            args = [core_utils.get_qualify_name(x,row['library'],aosp_class_map,gps_class_df) for x in args]
            args_fill = ','.join(args)
        qclass = core_utils.get_qualify_name(row['class'],row['library'],aosp_class_map,gps_class_df)
        qreturn = core_utils.get_qualify_name(row['return'],row['library'],aosp_class_map,gps_class_df)
        tmp = ['<',qclass,': ',qreturn,' ',row['method_name'],f"({args_fill})> -> ",f"_{row['classification'].upper()}_"]
        stmts.append(tmp)

    # TARGET FORMAT 
    # <class: return_type method_name(param1,...)> -> _SINK_
    # <class: return_type method_name(param1,...)> -> _SOURCE_    
    with open(output,'w') as f:
        for stmt in stmts:
            f.write(''.join(stmt) + '\n')
    