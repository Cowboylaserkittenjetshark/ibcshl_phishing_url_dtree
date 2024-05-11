import pandas as pd
import paths

data = pd.read_csv(paths.data.FILE)
target_label = 'phishing'
y = data[target_label]
X = data.drop(target_label, axis = 1)

extra_data = pd.read_csv(paths.data.EXTRA_FILE)

fetched_features = [
    'time_response',
    'domain_spf',
     'asn_ip',
     'time_domain_activation',
     'time_domain_expiration',
     'qty_ip_resolved',
     'qty_nameservers',
     'qty_mx_servers',
     'ttl_hostname',
     'tls_ssl_certificate',
     'qty_redirects',
     'url_google_index',
     'domain_google_index'
]

