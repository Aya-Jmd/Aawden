def read_record(records, actor, record_id):
    visible = {key: record for key, record in records.items() if record["owner_id"] == actor["id"]}
    return visible.get(record_id)
