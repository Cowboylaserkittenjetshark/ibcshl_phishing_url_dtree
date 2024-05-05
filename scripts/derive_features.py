import tldextract

def derive_features(X):
    X['tld'] = X.url.map(lambda url: tldextract.extract(url).suffix).rename('tld')
    X['is_https'] = X.url.map(lambda url: url.startswith('https://')).rename('is_https')
