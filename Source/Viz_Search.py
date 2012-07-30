import re
import datetime

# Using '__' to add a suffix later
common_date = r'(?P<cmonth__>0{,1}[0-9]|1[0-2])(?P<csep__>[-./])(?P<cday__>[0-2]{,1}[0-9]|3[0-1])(?P=csep__)(?P<cyear__>[0-9]{4}|[0-9]{2})'
iso_date = r'(?P<iyear__>[0-9]{4})(?P<isep__>[-./])(?P<imonth__>0[0-9]|1[0-2])(?P=isep__)(?P<iday__>[0-2][0-9]|3[0-1])'
dates = '(%s|%s)' % (iso_date, common_date)
unary_range = '(?P<query>after|since|before|until)\\s+%s' % (dates.replace('__', ''))
binary_range = '(?:from\\s+){,1}%s\\s+to\\s+%s' % (dates.replace('__', '1'), dates.replace('__', '2'))

date_prog = re.compile(dates.replace('__', ''))
unary_prog = re.compile(unary_range, re.I)
binary_prog = re.compile(binary_range, re.I)
today_prog = re.compile('today', re.I)

def extract_date(res, suffix=''):
    g = res.groupdict()
    year = g.get('iyear' + suffix)
    if year is None:
        year = g.get('cyear' + suffix)
    month = g.get('imonth' + suffix)
    if month is None:
        month = g.get('cmonth' + suffix)
    day = g.get('iday' + suffix)
    if day is None:
        day = g.get('cday' + suffix)
    day = int(day)
    month = int(month)
    year = int(year)

    if year < 70:
        year += 2000 
    elif year < 100:
        year += 1900
    return datetime.date(year, month, day)

def filter_metadata(metadata, search):
    res = binary_prog.match(search)
    if res is not None:
        date1 = extract_date(res, '1')
        date2 = extract_date(res, '2')
        return filter(lambda x: x.created_date >= date1 and x.created_date <= date2, metadata)

    res = unary_prog.match(search)
    if res is not None:
        query = res.group('query')
        date = extract_date(res)
        if query == 'before':
            return filter(lambda x: x.created_date < date, metadata)
        if query == 'after':
            return filter(lambda x: x.created_date > date, metadata)
        if query == 'until':
            return filter(lambda x: x.created_date <= date, metadata)
        return filter(lambda x: x.created_date >= date, metadata)
        
    res = date_prog.match(search)
    if res is not None:
        date = extract_date(res)
        return filter(lambda x: x.created_date == date, metadata)

    res = today_prog.match(search)
    if res is not None:
        date = datetime.datetime.now().date()
        return filter(lambda x: x.created_date == date, metadata)
    
    keywords = search.split()
    def kw_filter(x):
        for kw in keywords:
            if kw not in x.name:
                return False
        return True

    return filter(kw_filter, metadata)

if __name__ == '__main__':
    class TestData:
        def __init__(self, year, month, day, name):
            self.created_date = datetime.date(year, month, day)
            self.name = name

        def __repr__(self):
            return "DATA: '" + self.name + "' on " + str(self.created_date)

    data = [
        TestData(2012, 1, 1, "foo"),
        TestData(2012, 1, 2, "bar"),
        TestData(2012, 1, 3, "baz"),
        TestData(2012, 1, 4, "foo bar"),
        TestData(2012, 1, 5, "foo baz"),
        TestData(2013, 1, 5, "foo bar baz"),
        TestData(2011, 1, 5, "test"),
        TestData(2013, 1, 1, "test1"),
        TestData(2012, 7, 27, "today test1"),
        TestData(2012, 7, 28, "not today test1"),
    ]

    print filter_metadata(data, 'from 2012-01-02 to 01-02-13')
    print filter_metadata(data, '2012-01-02 to 01-02-13')
    print filter_metadata(data, 'since 2012-01-04')
    print filter_metadata(data, 'after 2012-01-04')
    print filter_metadata(data, 'before 2012-01-04')
    print filter_metadata(data, 'until 2012-01-04')
    print filter_metadata(data, '2012-01-04')
    print filter_metadata(data, 'today')
    print filter_metadata(data, 'foo')
    print filter_metadata(data, 'bar baz')
    print filter_metadata(data, 'test')
    print filter_metadata(data, 'bar')
