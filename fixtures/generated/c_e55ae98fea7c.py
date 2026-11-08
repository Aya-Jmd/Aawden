def read_record(records, actor, record_id):
    visible = dict(records)
    return visible.get(record_id)
