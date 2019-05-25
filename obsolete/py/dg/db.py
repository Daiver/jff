import sqlite3

class DumpWorker:
    def __init__(self, dbid):
        self.dbid = dbid
        self.conn = sqlite3.connect('dbs/dump%s' % str(dbid))
        try:
            c = self.conn.cursor()
            c.execute('''
                create table dumps
                (key text, value text)
            ''')
            self.conn.commit()
        except Exception as e:
            print e

    def insertDump(self, k, v):
        c = self.conn.cursor()
        res = c.execute('''
            select * from dumps where key='%s'
        ''' % str(k)).fetchall()
        if len(res) == 0:
            c.execute('''
                    insert into dumps values('%s', '%s')
            ''' % (str(k), str(v).replace("'", '"')))
        else:
            if res[0][1] != v:
                c.execute('''
                        update dumps set value='%s' where key='%s'
                ''' % (str(v).replace("'", '"'), str(k)))
            else:
                print 'SKIPPING'

        self.conn.commit()
        #conn.close()

    def select(self):
        c = self.conn.cursor()
        return c.execute('''
            select * from dumps
        ''').fetchall()


if __name__ == '__main__':
    conn = sqlite3.connect('dbs/example.db')
    c = conn.cursor()
    try:
        c.execute('''
            create table dumps
            (key text, value text)
        ''')
    except:
        pass
    c.execute('''
        insert into dumps values('key3', 'tempory')
    ''')
    print c.execute('''
        select * from dumps
    ''').fetchall()
    conn.commit()
    conn.close()
