def parseSeconds(seconds):
  SEC_PER_DAYS = 84600
  SEC_PER_HOURS = 3600
  SEC_PER_MINUTES = 60

  days = int(seconds // SEC_PER_DAYS)
  seconds = seconds % SEC_PER_DAYS

  hours = int(seconds // SEC_PER_HOURS)
  seconds = seconds % SEC_PER_HOURS

  minutes = int(seconds // SEC_PER_MINUTES)
  seconds = seconds % SEC_PER_MINUTES

  return f'{days}:{hours}:{minutes}:' + '{:.3f}'.format(seconds)

def pause():
  programPause = input("Press the <ENTER> key to continue...")