import argparse
import taint_specification
import aosp_crawler
import gps_crawler  
import pre_processing


parser = argparse.ArgumentParser('DocFlow main')
parser.add_argument('-op', '--operation', help="select the operation",
                    choices=['specification','search','classifier','sem_classifier']) 
parser.add_argument('-i','--input', help="file input path",required=False)
parser.add_argument('-o', '--output', help="file output path",required=False)
parser.add_argument('-cr','--crawler', help='' , 
                    choices=['aosp','libraries'],required=False)
parser.add_argument('-pr','--preprocessing', help='generates method representations',
                    choices=['FA','FB','FC','FD','FE','FF'],required=False)



if __name__ == "__main__":

    args = parser.parse_args()

    #TODO validate input and ouput format for each case
    input = args.input
    output = args.output 

    if args.operation is not None:

        if args.operation == 'specification':
            if input.endswith('.csv'):
                taint_specification.generate_taint_spec(args.input,args.output)
            else:
                print('Please enter a csv file an input')
        
        elif args.operation in ('search','classifier','sem_classifier'):
            print ('Classifiers not implemented. Please look at the scripts in /core') 
    
    elif args.crawler is not None:

        if args.crawler == 'aosp':
            # the default input file is in '../inputs/aosp_classes.txt'
            # ouput file with json format
            aosp_crawler.star_crawling(input,output)
        elif args.crawler == 'libraries':
            # the default input file is in '../inputs/gps-libraries.txt'
            # output file with json format
            gps_crawler.star_crawling(input,output)

    elif args.preprocessing is not None:

        pre_processing.generate_representations(input,output,args.preprocessing)

            
