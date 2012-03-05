#!/usr/bin/env python
import sys,os
'''
NAME
    feature_request.py

PURPOSE
    holding house for feature requests for the Payette project

INSTRUCTIONS
    using the template, file your feature request,.but first browse other users
    requests, as your request may already be filed.
'''

feature_requests = {}

feature_requests['template'] = {
    'requester':'Manny Spamboni',
    'email':'manny_spamboni@spaghetti_monster.com',
    'request':'so what do you want?',
    'completed':False}
feature_requests['callbacks'] = {
    'requester':'Tim Fuller',
    'email':'tjfulle@sandia.gov',
    'request':('python callbacks allowing for fortran models to '
               'communicate with Payette through python'),
    'completed':False}
feature_requests['rotation'] = {
    'requester':'Tim Fuller',
    'email':'tjfulle@sandia.gov',
    'request':('allow prescribed rotation in conjunction with '
               'prescribed strain as input'),
    'completed':False}
feature_requests['principal coords'] = {
    'requester':'Tim Fuller',
    'email':'tjfulle@sandia.gov',
    'request':('option to have full tensors on input converted to principal '
               'coordinates for calculation'),
    'completed':False}
feature_requests['restart'] = {
    'requester':'Tim Fuller',
    'email':'tjfulle@sandia.gov',
    'request':('restart capability if simulation quits midway through'),
    'completed':True,
    'comments':('For a simulation "sim_name", the restart file is '
                'sim_name.prf. To run the restart file, execute: '
                'runPayette sim_name.prf'),
    'completed on':'June 22, 2011',
    'completed by':'Tim Fuller'
    }

def fixlen(pad,msg):
    maxlen = 70
    pad = int(pad)
    details = ['']
    for x in msg.split(' '):
        if len(details[-1]) < maxlen:
            details[-1] += x+' '
        else:
            details.append('\n'+' '*pad+x+' ')
            pass
        continue
    return ' '.join(details)

if __name__ == '__main__':
    print('FEATURE REQUESTS:')
    for k in feature_requests:
        if k == 'template': continue
        feature_request = feature_requests[k]
        if not feature_request['completed']:
            print('Request:   %s'%(k))
            print('Requester: %s'%(feature_request['requester']))
            print('Details:   %s\n'%(fixlen(11,feature_request['request'])))
            pass
        continue

    print('\nCOMPLETED FEATURE REQUESTS')
    for k in feature_requests:
        if k == 'template': continue
        feature_request = feature_requests[k]
        if feature_request['completed']:
            print('Request:        %s'%(k))
            print('Requester:      %s'%(feature_request['requester']))
            print('Details:        %s'%(feature_request['request']))
            print('Completed on:   %s'%(feature_request['completed on']))
            print('Completed by:   %s'%(feature_request['completed by']))
            print('Comments:       %s\n'%(fixlen(16,feature_request['comments'])))
        continue

