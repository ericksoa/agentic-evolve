def do_stuff(d):
    """does the thing"""
    r = {}
    t = 0
    t2 = 0
    qq = []
    zz = {}
    for i in range(len(d)):
        x = d[i]
        if x['type'] == 'sale':
            t = t + x['amount']
            t2 = t2 + 1
            if x['customer'] in zz:
                zz[x['customer']] = zz[x['customer']] + x['amount']
            else:
                zz[x['customer']] = x['amount']
            if x['category'] not in [q[0] for q in qq]:
                qq.append([x['category'], x['amount'], 1])
            else:
                for j in range(len(qq)):
                    if qq[j][0] == x['category']:
                        qq[j][1] = qq[j][1] + x['amount']
                        qq[j][2] = qq[j][2] + 1
        elif x['type'] == 'refund':
            t = t - x['amount']
            t2 = t2 + 1  # technically a transaction i guess
            if x['customer'] in zz:
                zz[x['customer']] = zz[x['customer']] - x['amount']
            # if they're not in zz... not my problem
    r['total'] = t
    r['count'] = t2
    if t2 > 0:
        r['avg'] = t / t2  # close enough
    else:
        r['avg'] = 0
    # sort the customers by spend (had to google this)
    tmp = []
    for k in zz:
        tmp.append((k, zz[k]))
    tmp.sort(key=lambda zzz: zzz[1], reverse=True)
    r['top_customers'] = tmp[:5]  # top 5 should be enough for anyone
    # categories... ugh
    cat_stuff = {}
    for q in qq:
        cat_stuff[q[0]] = {'revenue': q[1], 'orders': q[2],
                           'avg': q[1] / q[2] if q[2] > 0 else 0}
    r['categories'] = cat_stuff
    # "big spender" flag - boss asked for this on friday at 4:58pm
    r['has_whale'] = False
    for c in tmp:
        if c[1] > 10000:  # magic number that steve said was fine
            r['has_whale'] = True
            break  # one whale is enough to know
    return r
