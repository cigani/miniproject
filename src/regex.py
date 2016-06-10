import re
dict = {}
dict[re.compile('actionname (\d+) (\d+)')] = method
dict[re.compile('differentaction (\w+) (\w+)')] = appropriate_method

def execute_method_for(str):
    #Match each regex on the string
    matches = (
        (regex.match(str), f) for regex, f in dict.iteritems()
    )

    #Filter out empty matches, and extract groups
    matches = (
        (match.groups(), f) for match, f in matches if match is not None
    )


    #Apply all the functions
    for args, f in matches:
        f(*args)