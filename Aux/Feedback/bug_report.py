#!/usr/bin/env python

'''
NAME
    bug_report.py

PURPOSE
    general holding house for bugs encountered in the Payette project.

INSTRUCTIONS
    using the template, file your bug report, but ook at the KNOWN ISSUES
    section first as your bug may already be known.
'''

known_issues, bug_reports = {}, {}
bug_reports['template'] = {
    'name':'Manny Spamboni',
    'email':'manny_spamboni@spaghetti_monster.com',
    'date':'Month Day, Year',
    'bug':'description of bug, how it was found, etc.',
    'supporting files':['input file','other files'],
    'comments':'any relevant comments',
    'resolved':False,
    'resolved by':'no one yet',
    'resolved on':'Month Day, Year'}


known_issues['SAGE'] = {
    'name':'Tim Fuller',
    'date':'June 20, 2011',
    'issue':('specifying -i sage to buildPayette to use sages built in '
             'Python interpreter leads to an ImportError when trying to '
             'import the material libraries generated by f2py.'),
    'email':'tjfulle@sandia.gov',
    'resolved':False,
    'resolved by':'no one yet',
    'resolved on':'Month Day, Year'}
known_issues['PRDEF'] = {
    'name':'Tim Fuller',
    'date':'June 20, 2011',
    'issue':('for a prescribed deformation gradient with any rotation Payette '
             'quits with message about not yet supporting rotation'),
    'email':'tjfulle@sandia.gov',
    'resolved':False,
    'resolved by':'no one yet',
    'resolved on':'Month Day, Year'}
known_issues['PYTHON3.X'] = {
    'name':'Tim Fuller',
    'email':'tjfulle@sandia.gov',
    'issue':('specifying -i python3.x to buildPayette to use python3.x as '
             'Payettes Python interpreter leads to an ImportError when '
             'trying to import the material libraries generated by f2py.'),
    'resolved':True,
    'resolved by':'Tim Fuller',
    'resolved on':'June 27, 2011'}
known_issues['CALLBACKS'] = {
    'name':'Tim Fuller',
    'email':'tjfulle@sandia.gov',
    'issue':('Python callbacks not working properly on darwin and possibly Linux. '
             'i.e., writing a python function "bombed" that is sent into '
             'kayenta results in a segfault'),
    'resolved':False,
    'resolved by':'no one yet',
    'resolved on':'Month Day, Year'}

bug_reports['stress control array indexing'] = {
    'name':'Tim Fuller',
    'email':'tjfulle@sandia.gov',
    'date':'June 21, 2011',
    'bug':('for stress controlled loading, a vector subscript array v is formed '
           'that contains the components for which the stress is prescribed.  If '
           'consecutive components of stress are prescribed (starting with 0) '
           'then the value of components of v also correspond to the indices of '
           'the prescribed stress, if they are not consecutively given, then the '
           'the value of the components of v do not.  Some of the algorithms for '
           'passing values of prescribed stress into the stress array were '
           'written assuming otherwise and had to be rewritten.'),
    'supporting files':None,
    'comments':None,
    'resolved':True,
    'resolved by':'Tim Fuller',
    'resolved on':'June 21, 2011'}

if __name__ == '__main__':
    print('BUG REPORTS:')
    for k in bug_reports:
        if k == 'template': continue
        bug_report = bug_reports[k]
        if not bug_report['resolved']:
            print('Issue:    %s'%(k))
            print('Reporter: %s'%(bug_report['name']))
            print('Details:  %s'%(bug_report['bug']))
            pass
        continue
    print('\nKNOWN ISSUES:')
    for k in known_issues:
        if k == 'template': continue
        known_issue = known_issues[k]
        if not known_issue['resolved']:
            print('Issue:    %s'%(k))
            print('Reporter: %s'%(known_issue['name']))
            print('Details:  %s\n'%(known_issue['issue']))
            pass
        continue
    pass

