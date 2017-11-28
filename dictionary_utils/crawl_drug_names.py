from bs4 import BeautifulSoup
import urllib2
import time
import httplib
import pickle

def get_other_names(url):
    
    list_of_names = []

    print "Getting trade names from : " + url
    
    try:
        page = urllib2.urlopen(url).read()
    except httplib.IncompleteRead, e:
        page = e.partial
    except:
        print "Retrying for url : " + url
        return get_other_names(url)

    soup = BeautifulSoup(page, 'html.parser')

    if soup.find('h1', id='firstHeading') is None:
        return list_of_names
    article_name = soup.find('h1', id='firstHeading').get_text()
    article_name_list = article_name.split("/")
    list_of_names.extend(article_name_list)
    # list_of_names.append(article_name)
    info_box = soup.find('table', ['infobox'])
    if info_box is None:
        return list_of_names

    items = info_box.find_all('td')

    for item in items:
        if item.th is None:
            continue
        if item.th.get_text().strip().lower() == "trade names":
            names = item.th.get_text().strip().lower()
            names = names.split(",")
            list_of_names.extend(names)
            return list_of_names

    return list_of_names


def scrap_link(link):


    drug_tuples = []

    links_to_crawl = []

    print "Visiting Link : " + link

    try:
        page = urllib2.urlopen(link).read()
    except urllib2.HTTPError as e:
        if e.code == 404:
            print "Caught exception " + str(e)
            return [], []

    soup = BeautifulSoup(page, 'html.parser')

    if len(soup.find_all('h2')) == 1:

        selector = "#mw-content-text > div > p:nth-of-type(5)"

        items = soup.select(selector)

        items = items[0].find_all('a')

        for item in items:
            if item.has_attr("href"):
                links_to_crawl.append(base_url + item['href'])
        print links_to_crawl
    else:
        headers = soup.find_all('h2')
        headers = headers[:-1]
        headers_3 = soup.find_all('h3')
        headers.extend(headers_3)
        for head in headers[:-1]:
            if head.has_attr('id'):
                continue
            if head.has_attr('class'):
                continue
            u_l = head.findNext('ul').find_all('li')
            # print u_l
            for l in u_l:
                if l.has_attr('class'):
                    break
                if l.has_attr('id'):
                    break
                if l.b is None:
                    continue
                if l.b.a is None:
                    continue

                if "does not exist" in l.b.a['title']:
                    drug_tuples.append( (l.b.a.get_text(), "") )
                
                else:
                    name = l.b.a.get_text()
                    url = base_url + l.b.a['href']

                    trade_names = get_other_names(url)

                    

                    if len(trade_names) == 0:
                        drug_tuples.append( (l.b.a.get_text(), "") )
                    else:
                        for t in trade_names:
                            if t.strip().lower() == "others":
                                continue
                            else:   
                                drug_tuples.append( (name, t.strip().lower()) )

    return links_to_crawl, drug_tuples

count = 0
base_url = "https://en.wikipedia.org"

main_page_url = base_url + "/wiki/List_of_drugs"

main_page = urllib2.urlopen(main_page_url).read()
count +=1
soup = BeautifulSoup(main_page, 'html.parser')

all_links_tag = soup.body.find_all("a", class_="mw-selflink selflink")[0].parent.findChildren()

links_to_crawl = []
# print type(all_links_tag)
# print len(all_links_tag)

for tag in all_links_tag:
    # print tag
    if tag.has_attr('href'):
        links_to_crawl.append(base_url + tag['href'])

links_crawled = []

drug_links = []

drug_names = []

drug_tuples = []

more_links = []

i = 0
size = 5
for link in links_to_crawl:
    count +=1
    if count >= size:
        size += 2
        print "Sleeping for 120 seconds: ..."
        time.sleep(120)


    drug_links, link_tuples = scrap_link(link)

    if len(link_tuples) != 0:
        drug_tuples.extend(link_tuples)

    if len(drug_links) != 0:
        more_links.extend(drug_links)

links_to_crawl = more_links

if len(more_links) != 0:

    for link in links_to_crawl:

        if count >= size:
            size += 2
            print "Sleeping for 120 seconds: ..."
            time.sleep(120)

        drug_links, link_tuples = scrap_link(link)

        drug_tuples.extend(link_tuples)

        more_links.extend(drug_links)

print "Remaining Links : " 
print more_links

with open("drug_tuples2.pkl", "wb") as fp:
    pickle.dump(drug_tuples, fp, pickle.HIGHEST_PROTOCOL)


