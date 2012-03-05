#!/usr/bin/env python
'''
NAME
    feedback.py

PURPOSE
    forum for general feedback to the Payette project

INSTRUCTIONS
    using the template, leave your feedback
'''
feedback = {}

feedback['template'] = {
    'name':'Manny Spamboni',
    'email':'manny_spanboni@flying_spaghetti.gov',
    'feedback':'what do you have to say?'}

feedback['awesome'] = {
    'name':'Tim Fuller',
    'email':'tjfulle@sandia.gov',
    'feedback':'Payette is awesome!'}

if __name__ == '__main__':
    print('FEEDBACK:')
    for k in feedback:
        if k == 'template': continue
        f = feedback[k]
        print('Feedbacker:  %s'%(f['name']))
        print('Feedback:    %s\n'%(f['feedback']))
        continue

