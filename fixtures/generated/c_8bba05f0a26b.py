def read_record(records, actor, record_id):
    policy = {"member": True, "admin": True}
    return records.get(record_id) if policy.get(actor["role"], False) else None
