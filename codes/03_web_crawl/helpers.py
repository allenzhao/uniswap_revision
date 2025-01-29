import pandas as pd


def fix_address(address: str) -> str:
    """
    Fix the address obtained from Dune.

    Returns the fixed address when the address start with `\\x` to 0x, and make it all lower case.

    Parameters
    ----------
    address : str
        The hash (address) to fix

    Returns
    -------
    str
        Lower case hash (address) that start with 0x

    See Also
    --------
    fix_address_df : Fix address for a pd.DataFrame
    """
    fixed_address = '0' + address[1:]
    return fixed_address.lower()


def fix_address_df(df: pd.DataFrame, *column_names: str) -> pd.DataFrame:
    """
    Fix columns of hashes to standard format in a DataFrame obtained from Dune.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame object that contains the columns we want to fix
    column_names: str
        The column names (in str) to fix. Order does not matter

    Returns
    -------
    pd.DataFrame
        Fixed Data Frame object

    See Also
    --------
    fix_address
    """
    for col in column_names:
        df[col] = df[col].str.replace("\\", "0", regex=False).str.lower()
    return df


def keep_cols(df: pd.DataFrame, cols_to_keep: [str]) -> pd.DataFrame:
    """
    Keeps only the columns in cols_to_keep of data frame df and drop the rest

    Parameters
    ----------
    df:pd.DataFrame
        DataFrame to operate on
    cols_to_keep:[str]
        columns to keep

    Returns
    -------
    pd.DataFrame
        DataFrame after dropping
    """
    return df.drop(df.columns.difference(cols_to_keep), axis=1)


def col_to_lower(df: pd.DataFrame, *column_names: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    df:pd.DataFrame
        DataFrame to operate on
    column_names:str
        column names that need to be converted to lower

    Returns
    -------
    pd.DataFrame
        DataFrame after operation
    """
    for col in column_names:
        df[col] = df[col].str.lower()
    return df
