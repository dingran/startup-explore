from __future__ import print_function
import colorama
import datetime
from bs4 import BeautifulSoup
import random

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException

colorama.init()

import time


def log_time(kind='general', color_str=None):
    if color_str is None:
        if kind == 'error' or kind.startswith('e'):
            color_str = colorama.Fore.RED
        elif kind == 'info' or kind.startswith('i'):
            color_str = colorama.Fore.YELLOW
        elif kind == 'overwrite' or kind.startswith('o'):
            color_str = colorama.Fore.MAGENTA
        elif kind == 'write' or kind.startswith('w'):
            color_str = colorama.Fore.CYAN
        elif kind == 'highlight' or kind.startswith('h'):
            color_str = colorama.Fore.GREEN
        else:
            color_str = colorama.Fore.WHITE

    print(color_str + str(datetime.datetime.now()) + colorama.Fore.RESET, end=' ')


def init_driver(driver_type='Chrome'):
    log_time('info')
    print('initiating driver: {}'.format(driver_type))
    if driver_type == 'Chrome':
        dr = webdriver.Chrome()
    elif driver_type.startswith('Pha'):
        dr = webdriver.PhantomJS()
    elif driver_type.startswith('Fi'):
        dr = webdriver.Firefox()
    else:
        assert False
    dr.set_window_size(1920, 600)
    dr.wait = WebDriverWait(dr, 5)
    dr.set_page_load_timeout(25)
    return dr


def quit_driver(dr):
    log_time('info')
    print('closing driver...')
    dr.quit()


def calc_pause(base_seconds=3., variable_seconds=5.):
    return base_seconds + random.random() * variable_seconds


def set_pause(kind=1, t=None):
    log_time('info')
    if t is not None:
        kind_str = 'specific'
    else:
        if kind == 5:
            kind_str = 'ultra long'
            t = calc_pause(base_seconds=1000, variable_seconds=1000)
        elif kind == 4:
            kind_str = 'very long'
            t = calc_pause(base_seconds=100, variable_seconds=100)
        elif kind == 3:
            kind_str = 'long'
            t = calc_pause(base_seconds=10, variable_seconds=10)
        elif kind == 2:
            kind_str = 'short'
            t = calc_pause(base_seconds=3., variable_seconds=3.)
        else:
            kind_str = 'very short'
            t = calc_pause(base_seconds=0.5, variable_seconds=1.5)

    print('{} pause: {}s...'.format(kind_str, t))

    time.sleep(t)


def load_url(driver=None, url=None, n_attempts_limit=3):
    """
    page loader with n_attempts
    :param driver: 
    :param url: 
    :param n_attempts_limit: 
    :return: 
    """
    n_attempts = 0
    page_loaded = False
    while n_attempts < n_attempts_limit and not page_loaded:
        try:
            driver.get(url)
            page_loaded = True
            log_time()
            print('page loaded successfully: {}'.format(url))
        except TimeoutException:
            n_attempts += 1
            log_time('error')
            print('loading page timeout', url, 'attempt {}'.format(n_attempts))
            set_pause(1)
        except:
            n_attempts += 1
            log_time('error')
            print('loading page unknown error', url, 'attempt {}'.format(n_attempts))
            set_pause(1)

    if n_attempts == n_attempts_limit:
        driver.quit()
        log_time('error')
        print('loading page failed after {} attempts, now give up:'.format(n_attempts_limit), url)
        return False

    return True


from tqdm import tqdm
from random import shuffle
import os
import sys
import hashlib

def main():
    company_list_file = 'company_list_AL.txt'
    with open(company_list_file) as f:
        s = f.readlines()
    company_list = [x.strip() for x in s]

    shuffle(company_list)
    driver = init_driver()

    for c in tqdm(company_list):
        try:
            print(c)
        except:
            continue

        fname = os.path.join('google_text', '{}_CB.txt'.format(c))
        hash_str = str(hashlib.sha1(fname.encode('utf-8')).hexdigest()[-16:])
        fname_backup = os.path.join('google_text', '{}_CB.txt'.format(hash_str))
        if os.path.exists(fname) or os.path.exists(fname_backup):
            print('company exists, skip')
        else:
            target_url = 'https://www.google.com/search?q={}%20crunchbase'.format(c)
            load_url(driver=driver, url=target_url)
            set_pause(2)
            page = driver.page_source
            soup = BeautifulSoup(page, 'lxml')
            text = ''
            try:
                text = soup.findAll('span', {'class': 'st'})[0].get_text()
                print(text)
            except:
                log_time('error')
                print('failed to parse page {}'.format(target_url))

            try:
                with open(fname, 'w', encoding='utf-8') as f:
                    pass
            except:
                log_time('error')
                print('possibly illegal filename {}'.format(fname))
                print('switching to backup filename {}'.format(fname_backup))
                fname = fname_backup

            with open(fname, 'w', encoding='utf-8') as f:
                try:
                    f.write(text)
                except:
                    log_time('error')
                    print('failed to write file {}'.format(fname))
                    f.write('')

    quit_driver(driver)

if __name__ == '__main__':
    main()
