from copy import deepcopy
from typing import Dict
import enum
import logging
import pickle
import numpy as np
import torch


class NamedArrayLoadingError(Exception):
    pass


class NamedArrayEncodingMethod(bytes, enum.Enum):
    PICKLE_DICT = b"0001"


logger = logging.getLogger("NamedArray")


def _namedarray_op(op):

    def fn(self, value):
        if not (isinstance(value, NamedArray) and  # Check for matching structure.
                getattr(value, "_fields", None) == self._fields):
            if not isinstance(value, NamedArray):
                # Repeat value for each but respect any None.
                value = tuple(None if s is None else value for s in self)
            else:
                raise ValueError('namedarray - set an item with a different data structure')
        try:
            xs = {}
            for j, ((k, s), v) in enumerate(zip(self.items(), value)):
                if s is not None and v is not None:
                    exec(f"xs[k] = (s {op} v)")
                else:
                    exec(f"xs[k] = None")
        except (ValueError, IndexError, TypeError) as e:
            print(s.shape, v.shape)
            raise Exception(f"{type(e).__name__} occured in {self.__class__.__name__}"
                            " at field "
                            f"'{self._fields[j]}': {e}") from e
        return NamedArray(**xs)

    return fn


def _namedarray_iop(iop):

    def fn(self, value):
        if not (isinstance(value, NamedArray) and  # Check for matching structure.
                getattr(value, "_fields", None) == self._fields):
            if not isinstance(value, NamedArray):
                # Repeat value for each but respect any None.
                value = {k: None if s is None else value for k, s in self.items()}
            else:
                raise ValueError('namedarray - set an item with a different data structure')
        try:
            for j, (k, v) in enumerate(zip(self.keys(), value.values())):
                if self[k] is not None and v is not None:
                    exec(f"self[k] {iop} v")
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(f"{type(e).__name__} occured in {self.__class__.__name__}"
                            " at field "
                            f"'{self._fields[j]}': {e}") from e
        return self

    return fn


def dumps(namedarray_obj, method="pickle"):
    if method == "pickle":
        return NamedArrayEncodingMethod.PICKLE_DICT.value + pickle.dumps(
            (namedarray_obj.__class__.__name__, namedarray_obj.to_dict()))
    else:
        raise NotImplementedError(
            f"Unknown method {method}. Available are {((m, m.value) for m in NamedArrayEncodingMethod)}")


def loads(b):
    if b[:4] == NamedArrayEncodingMethod.PICKLE_DICT.value:
        class_name, values = pickle.loads(b[4:])
        return from_dict(values=values)
    else:
        raise NotImplementedError(
            f"Unknown prefix {b[:4]}. Available are {((m, m.value) for m in NamedArrayEncodingMethod)}")


class NamedArray:
    """A class decorator modified from the `namedarraytuple` class in rlpyt repo,
    referring to
    https://github.com/astooke/rlpyt/blob/master/rlpyt/utils/collections.py#L16.

    NamedArray supports dict-like unpacking and string indexing, and exposes integer slicing reads
    and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).

    Note that namedarray supports nested structure.,
    i.e., the elements of a NamedArray could also be NamedArray.

    Example:
    >>> class Point(NamedArray):
    ...     def __init__(self,
    ...         x: np.ndarray,
    ...         y: np.ndarray,
    ...         ):
    ...         super().__init__(x=x, y=y)
    >>> p=Point(np.array([1,2]), np.array([3,4]))
    >>> p
    Point(x=array([1, 2]), y=array([3, 4]))
    >>> p[:-1]
    Point(x=array([1]), y=array([3]))
    >>> p[0]
    Point(x=1, y=3)
    >>> p.x
    array([1, 2])
    >>> p['y']
    array([3, 4])
    >>> p[0] = 0
    >>> p
    Point(x=array([0, 2]), y=array([0, 4]))
    >>> p[0] = Point(5, 5)
    >>> p
    Point(x=array([5, 2]), y=array([5, 4]))
    >>> 'x' in p
    True
    >>> list(p.keys())
    ['x', 'y']
    >>> list(p.values())
    [array([5, 2]), array([5, 4])]
    >>> for k, v in p.items():
    ...     print(k, v)
    ...
    x [5 2]
    y [5 4]
    >>> def foo(x, y):
    ...     print(x, y)
    ...
    >>> foo(**p)
    [5 2] [5 4]
    """

    def __init__(self, **kwargs):
        """

        Args:
            data: key-value following {field_name: otherNamedArray/None/np.ndarray/torch.Tensor}
        """
        self._fields = sorted(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __iter__(self):
        for k in self._fields:
            yield getattr(self, k)

    def __getitem__(self, loc):
        """If the index is string, return getattr(self, index).
        If the index is integer/slice, return a new dataclass instance containing
        the selected index or slice from each field.

        Args:
            loc (str or slice): Key or indices to get.

        Raises:
            Exception: To locate in which field the error occurs.

        Returns:
            Any: An element of the dataclass or a new dataclass
                object composed of the subarrays.
        """
        if isinstance(loc, str):
            # str indexing like in dict
            return getattr(self, loc)
        else:
            try:
                return self.__class__(**{s: None if self[s] is None else self[s][loc] for s in self._fields})
            except IndexError as e:
                for j, s in enumerate(self):
                    if s is None:
                        continue
                    try:
                        _ = s[loc]
                    except IndexError:
                        raise Exception(f"Occured in {self.__class__} at field "
                                        f"'{self._fields[j]}'.") from e

    def __setitem__(self, loc, value):
        """If input value is the same dataclass type, iterate through its
        fields and assign values into selected index or slice of corresponding
        field.  Else, assign whole of value to selected index or slice of
        all fields. Ignore fields that are both None.

        Args:
            loc (str or slice): Key or indices to set.
            value (Any): A dataclass instance with the same structure
                or elements of the dataclass object.

        Raises:
            Exception: To locate in which field the error occurs.
        """
        if isinstance(loc, str):
            setattr(self, loc, value)
        else:
            if not (isinstance(value, NamedArray) and  # Check for matching structure.
                    getattr(value, "_fields", None) == self._fields):
                if not isinstance(value, NamedArray):
                    # Repeat value for each but respect any None.
                    value = tuple(None if s is None else value for s in self)
                else:
                    raise ValueError('namedarray - set an item with a different data structure')
            try:
                for j, (s, v) in enumerate(zip(self, value)):
                    if s is not None and v is not None:
                        s[loc] = v
            except (ValueError, IndexError, TypeError) as e:
                raise Exception(f"{type(e).__name__} occured in {self.__class__.__name__}"
                                " at field "
                                f"'{self._fields[j]}': {e}") from e

    def __contains__(self, key):
        """Checks presence of field name (unlike tuple; like dict).

        Args:
            key (str): The queried field name.

        Returns:
            bool: Query result.
        """
        return key in self._fields

    def __getstate__(self):
        return {k: v for k, v in self.items()}

    def __setstate__(self, state):
        self.__init__(**state)

    def values(self):
        for v in self:
            yield v

    def keys(self):
        for k in self._fields:
            yield k

    def __len__(self):
        return len(self._fields)

    def length(self, dim=0):
        for k, v in self.items():
            if v is None:
                continue
            elif isinstance(v, (np.ndarray, torch.Tensor)):
                if dim < v.ndim:
                    return v.shape[dim]
                else:
                    continue
            else:
                continue
        else:
            raise IndexError(f"No entry has shape on dim={dim}.")

    def unique_of(self, field, exclude_values=(None,)):
        """Get the unique value of a field
        """
        unique_values = np.unique(self[field])
        unique_values = unique_values[np.in1d(unique_values, exclude_values, invert=True)]
        if len(unique_values) != 1:
            return None
        else:
            return unique_values[0]

    def average_of(self, field, ignore_negative=True):
        """Get the average value of the sample
        Returns:
            version: average version of the sample in trainer steps. None if no version is specified for any data.
        """
        values = self[field]
        if len(values) > 0:
            if ignore_negative:
                return np.nanmean(np.where(values >= 0, values, np.nan))
            else:
                return values.mean()
        else:
            return None

    def max_of(self, field, ignore_negative=True):
        """Get the average value of the sample
        Returns:
            version: average version of the sample in trainer steps. None if no version is specified for any data.
        """
        values = self[field]
        if len(values) > 0:
            if ignore_negative:
                return np.nanmax(np.where(values >= 0, values, np.nan))
            else:
                return values.max()
        else:
            return None

    def min_of(self, field, ignore_negative=True):
        """Get the average value of the sample
        Returns:
            version: average version of the sample in trainer steps. None if no version is specified for any data.
        """
        values = self[field]
        if len(values) > 0:
            if ignore_negative:
                return np.nanmin(np.where(values >= 0, values, np.nan))
            else:
                return values.min()
        else:
            return None

    def items(self):
        """Iterate over ordered (field_name, value) pairs.

        Yields:
            tuple[str,Any]: (field_name, value) pairs
        """
        for k, v in zip(self._fields, self):
            yield k, v

    def to_dict(self):
        result = {}
        for k, v in self.items():
            if isinstance(v, NamedArray):
                result[k] = v.to_dict()
            elif v is None:
                result[k] = None
            else:
                result[k] = v
        return result

    @property
    def shape(self):
        return recursive_apply(self, lambda x: x.shape).to_dict()

    def size(self):
        return self.shape

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(k+'='+repr(v) for k, v in self.items())})"

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    __add__ = _namedarray_op('+')
    __sub__ = _namedarray_op('-')
    __mul__ = _namedarray_op('*')
    __truediv__ = _namedarray_op('/')
    __iadd__ = _namedarray_iop('+=')
    __isub__ = _namedarray_iop('-=')
    __imul__ = _namedarray_iop('*=')
    __itruediv__ = _namedarray_iop('/=')
    __repr__ = __str__


def from_dict(values: Dict):
    """Create namedarray object from Nested Dict of arrays.
    Args:
        values: Nested key-value object of data. value should of type None, Numpy Array, or Torch Tensor.
                Return None if length of value is 0.
    Returns:
        NamedArray with the same data structure as input. If values is None, return None.
    Example:
    >>> a = from_dict({"x": np.array([1, 2]), "y": np.array([3,4])})
    >>> a.x
    array([1, 2])
    >>> a.y
    array([3, 4])
    >>> a[1:]
    NamedArray(x=[2],y=[4])
    >>> obs = {"state":{"speed": np.array([1, 2, 3]), "position": np.array([4, 5])}, "vision": np.array([[7],[8],[9]])}
    >>> obs_na = from_dict(obs)
    >>> obs_na
    NamedArray(state=NamedArray(position=[4 5],speed=[1 2 3]),vision=[[7]
     [8]
     [9]])
    >>> obs_na.state
    NamedArray(position=[4 5],speed=[1 2 3])
    """
    if values is None or len(values) == 0:
        return None
    for k, v in values.items():
        if isinstance(v, dict):
            values[k] = from_dict(v)
    return NamedArray(**values)


def array_like(x, value=0):
    if isinstance(x, NamedArray):
        return NamedArray(**{k: array_like(v, value) for k, v in x.items()})
    else:
        if isinstance(x, np.ndarray):
            data = np.zeros_like(x)
        else:
            assert isinstance(x, torch.Tensor), ('Currently, namedarray only supports'
                                                 f' torch.Tensor and numpy.array (input is {type(x)})')
            data = torch.zeros_like(x)
        if value != 0:
            data[:] = value
        return data


def __array_filter_none(xs):
    is_not_nones = [x is not None for x in xs]
    if all(is_not_nones) or all(x is None for x in xs):
        return
    else:
        example_x = xs[is_not_nones.index(True)]
        for i, x in enumerate(xs):
            xs[i] = array_like(example_x) if x is None else x


def recursive_aggregate(xs, aggregate_fn):
    """Recursively aggregate a list of namedarray instances.
    Typically recursively stacking or concatenating.

    Args:
        xs (List[Any]): A list of namedarrays or
            appropriate aggregation targets (e.g. numpy.ndarray).
        aggregate_fn (function): The aggregation function to be applied.

    Returns:
        Any: The aggregated result with the same data type of elements in xs.
    """
    __array_filter_none(xs)
    try:
        if isinstance(xs[0], NamedArray):
            return NamedArray(
                **{k: recursive_aggregate([x[k] for x in xs], aggregate_fn)
                   for k in xs[0].keys()})
        elif xs[0] is None:
            return None
        else:
            return aggregate_fn(xs)
    except Exception as e:
        raise ValueError(f"Namedarray aggregation failed. dtypes: {[type(x) for x in xs]}: {e}")


def recursive_apply(x, fn):
    """Recursively apply a function to a namedarray x.

    Args:
        x (Any): The instance of a namedarray subclass
            or an appropriate target to apply fn.
        fn (function): The function to be applied.
    """
    if isinstance(x, NamedArray):
        return NamedArray(**{k: recursive_apply(v, fn) for k, v in x.items()})
    elif x is None:
        return None
    else:
        return fn(x)
