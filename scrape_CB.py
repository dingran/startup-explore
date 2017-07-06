from scrape_google import *

company_list_file = 'company_list_CB.txt'
with open(company_list_file) as f:
    s = f.readlines()
company_list = [x.strip() for x in s]


target_url = 'https://www.crunchbase.com/organization/{}#/entity'
driver = init_driver()
shuffle(company_list)

for c in company_list:
    c_new = c.replace(' ', '-').replace(',', '-').replace('.', '-').replace('@', 'a').replace("'", '-')

    fname = os.path.join('crunchbase_text', '{}_CB.txt'.format(c_new))

    print(c, c_new)
    if os.path.exists(fname):
        print('company exists, skip')
    else:
        load_url(driver=driver, url=target_url.format(c_new))
        page = driver.page_source
        soup = BeautifulSoup(page, 'lxml')
        text = ''
        try:
            text = soup.findAll('div', {'class': 'description-ellipsis'})[0].get_text()
        except:
            log_time('error')
            print('failed to parse page')

        with open(fname, 'w', encoding='utf-8') as f:
            try:
                f.write(text)
            except:
                log_time('error')
                print('failed to write file {}'.format(fname))
                f.write('')

quit_driver(driver)
