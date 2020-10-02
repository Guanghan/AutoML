"""
Google style logger
"""
import glog as log

# Setting up level
log.setLevel("INFO")  # Integer levels are also allowed.
log.info("It works.")
log.warn("Something not ideal")
log.error("Something went wrong")
log.fatal("AAAAAAAAAAAAAAA!")

# Usage examples
obj1 = 1
obj2 = 2
condition = (obj1 != obj2)
log.check(condition)
log.check_eq(obj1, obj2)
log.check_ne(obj1, obj2)
log.check_le(obj1, obj2)
log.check_ge(obj1, obj2)
log.check_lt(obj1, obj2)
log.check_gt(obj1, obj2)
log.check_notnone(obj1, obj2)