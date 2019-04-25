import sets

class TokenScaner:
    def __init__(self, string):
        self.string = string
        self.index = 0
        self.symbols = '<>,./?\'"\\|=+-_)(*&^%$#@!`~;:[]{}'
        self.white   = ' \n\t'
        self.ssymbols = sets.Set()
        self.swhite = sets.Set()
        for x in self.symbols: self.ssymbols.add(x)
        for x in self.white: self.swhite.add(x)

    def empty(self): return self.index >= len(self.string)

    def getToken(self):
        while self.index < len(self.string) and self.string[self.index] in self.swhite :
            self.index += 1
        if self.index >= len(self.string): return None
        sflag = False
        if self.string[self.index] in self.symbols: sflag = True
        res = ''
        while (self.index < len(self.string) and self.string[self.index] not in self.swhite) and ((self.string[self.index] in self.ssymbols) == sflag):
            res += self.string[self.index]
            self.index += 1
        return res

if __name__ == '__main__':
    ts = TokenScaner(u'<html>, not ,rgjiejrigj*90-(10 - 78)')
    while not ts.empty():
        print ts.getToken()
