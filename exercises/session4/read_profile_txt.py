import pstats
p = pstats.Stats('profile.txt')
p.sort_stats('tottime').print_stats(30)