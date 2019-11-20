# ---------------------------BEGIN-HEADER------------------------------------
# Copyright (2016) Sandia Corporation. 
# Under the terms of Contract DE-AC04-94AL85000 
# with Sandia Corporation, the U.S. Government 
# retains certain rights in this software. This 
# software is distributed under the GNU General 
# Public License.

# FitSNAP.py - A Python framework for fitting SNAP interatomic potentials

# Original author: Aidan P. Thompson, athomps@sandia.gov
# http://www.cs.sandia.gov/~athomps, Sandia National Laboratories
# Key contributors:
# Mary Alice Cusentino
# Adam Stephens
# Mitchell Wood

# Additional authors: 
# Elizabeth Decolvenaere
# Stan Moore
# Steve Plimpton
# Gary Saavedra
# Peter Schultz
# Laura Swiler

# ----------------------------END-HEADER-------------------------------------

import sys
from snapexception import SNAPException

class InFileException(SNAPException):
    pass

class _Keyword(object):
    def __init__(self,keyword=None,default=None,vtype=None,range=None,
            aliases=None,required=None):
        self._name=keyword
        self.default=default
        self.vtype=vtype
        self._range_check=range
        self.aliases = aliases
        self._required_check = required
        # basic error checking
        # Make sure aliases is of type set, and add keyword to its
        # own set of permitted aliases.
        if self.aliases is None:
            self.aliases = set()
        elif type(self.aliases) is not set:
            raise InFileException("Error: aliases for keyword %s " + \
                    str(self.type) + " is not of required type set.")
        self.aliases.add(self._name)
        # If default is provided, make sure it is of an allowed type.
        # If vtype is also provided, make sure it matches type(default)
        if self.default != None:
            if type(self.default) not in set((int,float,str)):
                raise InFileException("Error: Type of default for keyword " + \
                        str(self._name) + " must be int, float, or str")
            if self.vtype and type(self.default) != self.vtype:
                raise InFileException("Error: Type of default for keyword " + \
                        str(self._name) + " does not match declared type")
            else:
                self.vtype = type(default)
        elif self.vtype not in set((int,float,str)):
            raise InFileException("Error: Attempted to add an input file " + \
                    "keyword of type " + str(self.vtype) + ". Permitted " + \
                    "types are int, float, and str.")

    def check_range(self,v):
        if self._range_check:
            return self._range_check(v)
        else:
            return None # always None if no range checking defined

    def verify_requirements(self,defined):
        if self._required_check:
            return self._required_check(defined)
        return None


class InFileParser(object):
    def __init__(self):
        self._declared={}

    def add_keyword(self,keyword=None,default=None,vtype=None,range=None,
            aliases=None, required=None):
        
        self._declared[keyword] = _Keyword(keyword,default,vtype,range,aliases,
                required)
    
    def parse(self,file=None):
        if file:
            try:
                fp = open(file,"r")
                raw = fp.readlines()
                fp.close()
            except IOError:
                raise InFileException("Error: Attempt to open input file %s " + \
                        file + " failed.")
        else: # read from stdin
            raw = sys.stdin.readlines()
        # Convert user input into a dictionary
        userKeys = self._raw_to_dict(raw)
        # verify that keywords from file are valid keywords
        self._check_for_definition(userKeys)
        # convert to appropriate types
        self._convert_types(userKeys)
        # Verify that all keywords are within valid ranges
        self._check_ranges(userKeys)
        # Create a separate dictionary of defaults and combine them with
        # userKeys
        allKeys = self._set_defaults()
        allKeys.update(userKeys)
        # Verify that all required keywords are present.
        self._verify_requirements(allKeys)
        # everything is good!
        return allKeys


    def _set_defaults(self):
        defaults = {}
        for k, v in self._declared.iteritems():
            if v.default != None:
                defaults[k] = v.default
        return defaults

    def _raw_to_dict(self,raw):
        # remove newlines and comments
        cleaned = [ line.strip().split('#')[0] for line in raw]
        # remove empties
        cleaned = [ line for line in cleaned if line]
        # split into tokens: ['keyword','value'].
        tokenized = [ line.split('=',1) for line in cleaned ]
        # make sure each "line" is now a list of length 2, and strip
        # surrounding whitespace
        for i, token in enumerate(tokenized):
            if len(token) != 2:
                raise InFileException("Error: Input file syntax error.")
            tokenized[i] = [t.strip() for t in token]
        # create dictionary
        keyDict = {}
        for token in tokenized:
            keyDict[token[0]] = token[1]
        return keyDict


    def _check_for_definition(self,userKeys):
        for uk in userKeys.keys():
            found = False
            for dk, v in self._declared.iteritems():
                if uk in v.aliases:
                    found = True
                    break
            if not found:
                raise InFileException("Error: keyword %s in input " % uk +\
                        "file is not recognized.")

    def _convert_types(self,userKeys):
        for k, v in userKeys.iteritems():
            try:
                userKeys[k] = self._declared[k].vtype(v)
            except ValueError:
                raise InFileException("Error: Type conversion error for " + \
                        "keyword %s in input file " % k)

    def _check_ranges(self,userKeys):
        for k, v in userKeys.iteritems():
            message = self._declared[k].check_range(v)
            if message:
                raise InFileException("Error: Range: %s" % message)



    def _verify_requirements(self,allKeys):
       for k, v in self._declared.iteritems():
           message = v.verify_requirements(allKeys)
           if message:
               raise InFileException("Error: Requirements: %s " % message)




