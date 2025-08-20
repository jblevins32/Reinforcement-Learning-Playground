def units(x):
    '''Convert seconds to a human-readable format.'''
    if x > 3600*24:
        return f"{x / (3600*24):.2f} days"
    if x > 3600:
        return f"{x / 3600:.2f} hr"
    if x > 60:
        return f"{x / 60:.2f} min"
    else:
        return f"{x:.2f} sec"