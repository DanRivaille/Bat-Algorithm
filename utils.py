def parseSeconds(seconds):
  SEC_PER_DAYS = 84600
  SEC_PER_HOURS = 3600
  SEC_PER_MINUTES = 60

  days = seconds // SEC_PER_DAYS
  seconds = seconds % SEC_PER_DAYS

  hours = seconds // SEC_PER_HOURS
  seconds = seconds % SEC_PER_HOURS

  minutes = seconds // SEC_PER_MINUTES
  seconds = seconds % SEC_PER_MINUTES

  return f'{days}:{hours}:{minutes}:' + '{:.3f}'.format(seconds)