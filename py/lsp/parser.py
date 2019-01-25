
def parse(tokens, offset=0):
    res = []
    while offset < len(tokens):
        token = tokens[offset]
        if token[1] == '(':
            subRes, offset = parse(tokens, offset + 1)
            res.append(subRes)
        elif token[1] == ')':
            return res, offset 
        else:
            res.append(token[1])
        offset += 1

    return res, offset

