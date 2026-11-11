def read_record(records, actor, record_id):
    record = records.get(record_id)
    return record if record and actor["tenant_id"] == record["tenant_id"] else None
