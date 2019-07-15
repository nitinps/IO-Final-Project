import csv
import os 
import requests
from bs4 import BeautifulSoup
url='https://karki23.github.io/Weather-Data/assignment.html'
sauce=requests.get(url)
srccode=BeautifulSoup(sauce.content, "html.parser")
city=srccode.find_all('a')
os.mkdir("Weather info")
for i in city:
    s=i.get('href')[0:len(i)-5:]
    url1='https://karki23.github.io/Weather-Data/'+i.get('href')
    sauce1=requests.get(url1)
    soup=BeautifulSoup(sauce1.content, "html.parser")
    r=soup.find_all('tr')
    r.pop(0) 
    file_name="Weather info\\"+s+"csv"
    f=open(file_name, "w", newline="")
    title=soup.find_all('th')
    title1=[i.text for i in title]
    write=csv.writer(f)
    write.writerow(title1)
    for i in r:    
        c=i.find_all('td')
        c1=[j.text for j in c]
        write.writerow(c1)
    f.close()
