"""Filename and naming helpers."""


def extract_campaign_names(file, extension='_v2.nc'):
    """Extract a campaign name from a project filename."""
    file_without_extension = file.rsplit(extension, 1)[0]
    pre_version = file_without_extension.split('-')[0]
    parts = pre_version.split('_')

    if 'Nortek' in parts and 'Burst' in parts:
        campaign_name = parts[3]
    else:
        campaign_name = parts[-1]
        if 'WorkHorse' in campaign_name:
            campaign_name = parts[-2]

    return campaign_name
