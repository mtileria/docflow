"""
This module crawls Google Play Services APIs documentation from
https://developers.google.com/android/reference/packages.html
and save the output as json file.
"""
import time
import json 
from bs4 import BeautifulSoup
import requests
import queue


class LibraryParser:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.classes = []
    
    def convert_json(self):
        content = {
            'name':self.name,
            'path':self.path,
            'classes': [x.to_dict() for x in self.classes]
        }
        return content #json.dumps(content)

class ClazzParser:
    def __init__(self,name, type_obj, link, des):
        self.name  = name
        self.type_obj = type_obj
        self.link = link
        self.desciption = des
        self.methods = []
    
    def __str__(self) -> str:
        return f'{self.name} : {len(self.methods)}'

    def to_dict(self):
        return {'name':self.name,'type_obj':self.type_obj,'link':self.link,
        'description':self.desciption,'methods':[x.to_dict() for x in self.methods]}

class MethodParser:
    def __init__(self,name, return_type, des):
        self.name  = name
        self.return_type = return_type
        self.desciption = des
    
    def to_dict(self):
        return {'name':self.name,'return_type':self.return_type,'description':self.desciption}

def get_soup(url,full_url=True):
    if not full_url:
        base_url = 'https://developers.google.com/'
        url = ''.join([base_url,url])
    cookies = {'required_cookie': 'required_value'}
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, cookies=cookies, headers=headers)
    soup = BeautifulSoup(response.text,features="html.parser") 
    return soup

def parse_summary(soup):
    classes = []
    for inter_class in ['classes','interfaces']:
        h3 = soup.find("h3", attrs={"id": inter_class})
        if h3:
            i_table = h3.find_next()
            for tr in i_table.find_all('tr'):
                for td in tr.find_all('td'):
                    if td['class'][0] == 'jd-linkcol':
                        href = td.find_all('a')[0]['href']
                        name = td.get_text(strip=True)
                    if td['class'][0] == 'jd-descrcol':
                        desc = td.text.replace('\n','').strip('"')
                        desc = ' '.join(desc.split())     
                tmp_class = ClazzParser(name,inter_class,href,desc)
                classes.append(tmp_class)
    return classes

def parse_pub_methods(soup):
    methods = []
    pmethods  = soup.find('table', attrs={"id": "pubmethods"})
    if pmethods:
        for tr in pmethods.find_all('tr'):
            for td in tr.find_all('td'):
                if td['class'][0] == 'jd-typecol':
                    return_type = ''.join(list(td.stripped_strings))
                elif td['class'][0] == 'jd-linkcol':
                    sp = td.text.replace('\n','').strip('"') 
                    sp = ' '.join(sp.split())     
                    desc_long = sp.split(')') # name and descripton
                    name = desc_long[0]+')'
                    description = desc_long[1].lstrip()
            methods.append(MethodParser(name,return_type,description))
    return methods


def get_nested_classes(soup, url):
    class_base = url.split('/')[-1]
    nested_data = [] 
    nested_table = soup.find('table', attrs={"id": "nestedclasses"})
    if nested_table:
        rows = nested_table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            relevant = True
            for td in cols:
                if td['class'][0] == 'jd-typecol':
                    if td.get_text().startswith('@'): 
                        relevant = False
                    class_inter = td.get_text(strip=True)
                elif td['class'][0] == 'jd-linkcol':  
                    class_name = td.text.strip()
                    link = url.replace(class_base,'') + class_name
                elif td['class'][0] == 'jd-descrcol':
                    sp = td.text.replace('\n','').strip('"')
                    description = ' '.join(sp.split())     
            if relevant:
                tmp = ClazzParser(class_name,class_inter,link,description)
                nested_data.append(tmp)
    return nested_data


def star_crawling(input,output):
    start = time.time()
    base_url = 'https://developers.google.com'
    with open (input, 'r') as file_in:
        urls = file_in.read().splitlines()

    libraries_list = []
    for url in urls:
        lib_name = url.split('/')[-2]
        errors = []
        url = base_url + url
        library = LibraryParser(lib_name,url)
        soup = get_soup(url)
        classes_preliminary = parse_summary(soup)
        library.classes = classes_preliminary
        # Load the queue with initial classes
        q1 = queue.Queue()
        for clazz in library.classes:
            q1.put(clazz)
        while not q1.empty():
            try:
                clazz = q1.get()
                # Step 1: parse the public methods from class doc
                soup = get_soup(clazz.link,full_url=False)
                tmp = parse_pub_methods(soup)
                clazz.methods.extend(tmp)
                # Step 2: parse nested classes and add them to the queue
                nested = get_nested_classes(soup,clazz.link)
                if len(nested) > 0:
                    library.classes.extend(nested)
                    for clazz in nested:
                        q1.put(clazz)
            except Exception as e:
                print(e)
                errors.append(clazz.link)
    
        # serialize library data
        lib_dict = library.convert_json()
        libraries_list.append(lib_dict)
    
    # Finally write libraries to json
    json_string = json.dumps(libraries_list)
    with open(output, 'w') as outfile:
        outfile.write(json_string)
    # write errors
    if len(errors) > 0:
        for item in errors:
            print(item)

    end = time.time()
    sec = end - start
    print(f'Processing took {sec} seconds')




