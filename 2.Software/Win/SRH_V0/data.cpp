#include "data.h"

Data::Data(QObject *parent) : QObject(parent)
{

}

QByteArray Data::packRXZ(int16_t R, int16_t X)
{
    QByteArray ba;
    ba.resize(8);
    quint8 j;
    j=0;
    ba[j++]=Head0;//包头
    ba[j++]=Head1;//包头
    short2byte(R, &ba, j);
    j += 2;
    short2byte(X, &ba, j);
    j += 2;
    ba[j++]=Tail0;//包头
    ba[j++]=Tail1;//包头
    return ba;
}

void Data::short2byte(short n, QByteArray *ba, int m)
{
    (*ba).data()[m+0] = (n>>8);
    (*ba).data()[m+1] = (n);
}
