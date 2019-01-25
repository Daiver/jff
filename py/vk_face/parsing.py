import os
import time
import socks
import socket
import mechanize
import cookielib
import pickle
from mechanize import Browser
#import lxml
#from lxml import etree
import StringIO

def doTorProxy():
    def create_connection(address, timeout=None, source_address=None):
        sock = socks.socksocket()
        sock.connect(address)
        return sock

    socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, "127.0.0.1", 9050)
    # patch the socket module
    socket.socket = socks.socksocket
    socket.create_connection = create_connection

def resetTor():
    os.system('service tor restart')

def testIP():
    import urllib2
    print 'ip', urllib2.urlopen('http://icanhazip.com').read()

def getLoginInfo():
    import getpass
    return (raw_input('email>'), getpass.getpass(), raw_input('phone>'))

def lockNloadBrowser():
    br = Browser()
    cj = cookielib.LWPCookieJar()
    br.set_cookiejar(cj)
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)

    # Follows refresh 0 but not hangs on refresh > 0
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
    br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
    return br


def logInToVk(br, data, type_phone=True):
    email, password, phone = data
    
    print 'entering vk'
    br.open('https://m.vk.com/').read()
    br.select_form(nr=0)
    br['email'] = email
    br['pass'] = password
    print 'loging'
    br.submit().read()
    if type_phone:
        try:
            br.select_form(nr=0)
            br['code'] = phone
            br.submit().read()
        except Exception as e:
            print 'Cannot type phone'
            print e

def saveBrowserCookies(br, f):
    cookiestr=''
    for c in br._ua_handlers['_cookies'].cookiejar:
        cookiestr+=c.name+'='+c.value+';'
    f.write(cookiestr)
    f.close()

def loadBrowserCookies(br, f):
    cookiestr = f.read()
    f.close()
    br.open('http://vk.com')
    while len(cookiestr)!=0:
        br.set_cookie(cookiestr)
        cookiestr=cookiestr[cookiestr.find(';')+1:]

def loadVkBrowserOrAuth(fname):
    br = lockNloadBrowser()
    if os.path.isfile(fname):
        loadBrowserCookies(br, open(fname))
        return br
    auth_data = getLoginInfo()
    logInToVk(br, auth_data)
    saveBrowserCookies(br, open(fname, 'w'))
    return br

def extractUserData(br, user_page):
    res = {}
    avatar_tag = '"profile_photo_link" href="'
    html = br.open(user_page).read()
    index = html.find(avatar_tag)
    if index == -1:
        print 'ERROR Index is -1'
        return None
    slice = html[index + len(avatar_tag):]
    res['avatar_url'] = slice[1:slice.find('"')]
    
    title_anchor = '<title>'
    index = html.find(title_anchor)
    res['utitle'] = html[index + len('title_anchor') : html.find('</title>', len(title_anchor) + index)]
    #slice = slice[:slice.find('onclick')]
    return res

def extractAvatarFromPage(br, user_page, avatar_url):
    html = br.open('%s?z=%s' % (user_page, avatar_url)).read()
    avatar_anchor = 'z_:["'
    lindex = html.find(avatar_anchor)
    if lindex == -1:
        print 'ERROR lIndex is -1'
        return None
    rindex = html.find('"', lindex + len(avatar_anchor))
    if rindex == -1:
        print 'ERROR rIndex is -1'
        return None
    return html[lindex + len(avatar_anchor):rindex]

def borderedSlice(s, start, finish, get_indexes=False):
    def check(index, txt): 
        if index == -1:
            raise Exception(txt)
    lindex = s.find(start) + len(start)
    check(lindex, 'bad lindex')
    rindex = s.find(finish, lindex)
    check(lindex, 'bad rindex')
    if get_indexes:
        return (s[lindex:rindex], lindex, rindex)

    return s[lindex:rindex]

def extractUserDataMobi(br, user_page):
    res = {'user_page' : user_page}
    profile_panel_anchor_st = '<div class="owner_panel profile_panel">'
    profile_panel_anchor_fh = '<div class="pp_cont">'
    html = br.open('https://m.vk.com/%s' % user_page).read()
    profile_panel = borderedSlice(html, profile_panel_anchor_st, profile_panel_anchor_fh)
    avatar_href_slice = borderedSlice(profile_panel, '<a href="/', '">')[len('album'):]
    res['uid'] = borderedSlice(avatar_href_slice, '', '_')
    return res

#https://m.vk.com/album85237080_0?rev=1&from=profile
def extractAvatarsListMobi(br, uid):
    st_anch = '<div class="photos_page thumbs_list">'
    fn_anch = '</div>'
    html = br.open('https://m.vk.com/album%s_0?rev=1&from=profile' % uid).read()
    data_to_analyse = borderedSlice(html, st_anch, fn_anch)
    res = []
    while len(data_to_analyse) > 0:
        try:
            url, l, r = borderedSlice(data_to_analyse, 'data-src_big="', '|', True)
            res.append(url)
            if l < 0 or r < 0 : break
            data_to_analyse = data_to_analyse[r:]
        except Exception as e:
            print e
            break
    return filter(lambda x: len(x) > 0, res)

def extractFriendsMobi(br, uid):
    html = br.open('https://m.vk.com/friends?id=%s' % uid).read()
    open('tmp', 'w').write(html)
    data_to_analyse = borderedSlice(html, '"container":"fr_search_items","field":"fr_search_field","btn":"fr_search_btn","clear_btn":false,"top_items":[', ']')
    uids = data_to_analyse.replace('"', "").split(',')
    return uids

#doTorProxy()
#testIP()
#resetTor()
#testIP()
fname = '/home/daiver/MyBrowser.dump'
br = loadVkBrowserOrAuth(fname)
start_time = time.time()
#print extractFriendsMobi(br, '93837391')
#exit()
#print br.open('http://vk.com').read()

user_pool = [
            #'id11099250', 'dark_daiver', 'kumaks'
            #'id19748148',
            #'id15062030', 'id130757186', 'id32629163', 'id7341431', 'ashaman'
        ]

from pool import tmp_pool

user_pool = map(lambda x: 'id' + x, tmp_pool)
#user_pool.sort()
print user_pool

import time
import random

watched = {}

data = []
try:
    for i, user_name in enumerate(user_pool):
        if user_name in watched: continue
        try:
            time.sleep(0.5 + 1.5 * random.random())
            print user_name, i + 1, '/', len(user_pool)
            uid = extractUserDataMobi(br, user_name)['uid']
            time.sleep(0.5 + 1.5 * random.random())
            avatars = extractAvatarsListMobi(br, uid)
            data.append((uid, avatars))
        except Exception as e:
            print uid, e
        watched[user_name] = 1
except KeyboardInterrupt as ki:
    print 'KI', ki

print data

dump_dir = '/home/daiver/dumps/vk_face'
for uid, avatars in data:
    print uid
    dir_name = os.path.join(dump_dir, uid, 'raw')
    os.system('mkdir -p %s' % dir_name)
    with open(os.path.join(dir_name, 'av_urls.txt'), 'w') as f:
        for x in avatars:
            f.write('%s\n' % x)
    for i, x in enumerate(avatars):
        #time.sleep(1)
        os.system('wget %s --output-document=%s/%d.jpg' % (x, dir_name, i))


saveBrowserCookies(br, open(fname, 'w'))
print 'finish', time.time() - start_time
'''
print 'avatar...'
user_page = 'https://vk.com/dark_daiver'
user_page = 'https://vk.com/id73476433'
user_page = 'https://vk.com/id10905716'
user_page = 'https://vk.com/kselon'
d1 = extractUserDataMobi(br, 'dark_daiver')
d2 = extractUserDataMobi(br, 'kselon')
extractAvatarsListMobi(br, d1['uid'])
extractAvatarsListMobi(br, d2['uid'])'''
'''uinfo = extractUserData(br, user_page)
print 'info', uinfo
print uinfo['utitle']
print extractAvatarFromPage(br, user_page, uinfo['avatar_url'])+'.jpg'
'''
#with open('tmp', 'w') as f:
#    f.write(html)
#print br.open('http://vk.com').read()


#br = Browser()
#print br.open('http://icanhazip.com').read()


