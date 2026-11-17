"""Enumerate DirectShow camera identities; indices match CAP_DSHOW enumeration."""
import ctypes as c
import uuid

def cameras():
    class GUID(c.Structure):_fields_=[('bytes',c.c_ubyte*16)]
    def guid(s):return GUID.from_buffer_copy(uuid.UUID(s).bytes_le)
    def method(ptr,index,restype,*args):
        return c.WINFUNCTYPE(restype,c.c_void_p,*args)(c.cast(ptr,c.POINTER(c.POINTER(c.c_void_p))).contents[index])
    def release(p):
        if p:method(p,2,c.c_ulong)(p)
    ole=c.OleDLL('ole32');aut=c.OleDLL('oleaut32')
    hr=ole.CoInitialize(None);initialized=hr in (0,1)
    dev=c.c_void_p();enum=c.c_void_p();result=[]
    try:
        cls=guid('62BE5D10-60EB-11D0-BD3B-00A0C911CE86');iid=guid('29840822-5B84-11D0-BD3B-00A0C911CE86')
        ole.CoCreateInstance(c.byref(cls),None,1,c.byref(iid),c.byref(dev))
        category=guid('860BB310-5D01-11D0-BD3B-00A0C911CE86')
        status=method(dev,3,c.c_long,c.POINTER(GUID),c.POINTER(c.c_void_p),c.c_ulong)(dev,c.byref(category),c.byref(enum),0)
        if status!=0:return result
        while True:
            mon=c.c_void_p();fetched=c.c_ulong()
            if method(enum,3,c.c_long,c.c_ulong,c.POINTER(c.c_void_p),c.POINTER(c.c_ulong))(enum,1,c.byref(mon),c.byref(fetched))!=0:break
            bag=c.c_void_p();props={}
            try:
                iidbag=guid('55272A00-42CB-11CE-8135-00AA004BB851')
                method(mon,9,c.c_long,c.c_void_p,c.c_void_p,c.POINTER(GUID),c.POINTER(c.c_void_p))(mon,None,None,c.byref(iidbag),c.byref(bag))
                for key in ['FriendlyName','DevicePath']:
                    variant=c.create_string_buffer(24)
                    code=method(bag,3,c.c_long,c.c_wchar_p,c.c_void_p,c.c_void_p)(bag,key,c.byref(variant),None)
                    if code==0 and c.c_ushort.from_buffer(variant).value==8:
                        ptr=c.c_void_p.from_buffer(variant,8).value
                        props[key]=c.wstring_at(ptr) if ptr else ''
                    aut.VariantClear(c.byref(variant))
                result.append(dict(index=len(result),**props))
            finally:release(bag);release(mon)
        return result
    finally:
        release(enum);release(dev)
        if initialized:ole.CoUninitialize()
