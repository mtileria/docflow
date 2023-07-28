import core_utils
import pandas as pd

function_map = {
    'FA':'generate_docs_bert_A',
    'FB':'generate_docs_bert_B',
    'FC':'generate_docs_bert_C',
    'FD':'generate_docs_bert_D',
    'FE':'generate_docs_bert_E',
    'FF':'generate_docs_bert_F'
}

def generate_representations(input, output, format):
    
    df = pd.read_csv(input)
    
    if format == 'FA':
        out = core_utils.generate_docs_FA(df) 
    if format == 'FB':
        out = core_utils.generate_docs_FB(df) 
    if format == 'FC':
        out = core_utils.generate_docs_FC(df) 
    if format == 'FD':
        out = core_utils.generate_docs_FD(df) 
    if format == 'FE':
        out = core_utils.generate_docs_FE(df)
    if format == 'FF':
        out = core_utils.generate_docs_FF(df)
    if format == 'FG':
        out = core_utils.generate_docs_FG(df)

    out_df = pd.DataFrame(out,columns=['description'])
    out_df['key'] =  df['class'] + '_' + df['method_name']
    out_df.to_csv(output)

    


